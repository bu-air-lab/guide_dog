import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset

from guide_dog_ppo.modules import ActorCritic
from guide_dog_ppo.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 base_velocity_estimator,
                 force_estimator,
                 isForceEstimator=True, 
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=0.01,
                 device='cpu',
                 ):

        self.device = device

        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate

        # PPO components
        self.actor_critic = actor_critic
        self.actor_critic.to(self.device)
        self.storage = None # initialized later
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=learning_rate)
        self.transition = RolloutStorage.Transition()

        # Base velocity estimator
        self.base_velocity_estimator = base_velocity_estimator
        self.base_velocity_estimator.to(self.device)
        self.base_velocity_estimator_loss = nn.MSELoss()
        self.base_velocity_estimator_optimizer = optim.Adam(self.base_velocity_estimator.parameters(), lr=learning_rate)

        # Force estimator
        self.force_estimator = force_estimator
        self.force_estimator.to(self.device)
        self.force_estimator_loss = nn.MSELoss()
        self.force_estimator_optimizer = optim.Adam(self.force_estimator.parameters(), lr=learning_rate)

        self.isForceEstimator = isForceEstimator

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss

    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape, self.device)

    def test_mode(self):
        self.actor_critic.test()
    
    def train_mode(self):
        self.actor_critic.train()

    def act(self, obs, critic_obs):

        if self.actor_critic.is_recurrent:
            self.transition.hidden_states = self.actor_critic.get_hidden_states()


        #Update obs with estimated state (replace features at the end of obs)
        estimated_base_vel = self.base_velocity_estimator(obs)


        if(self.isForceEstimator):

            #Estimating force requires past observations
            #self.storage.observations is num states x num envs x state size
            #Take observations in order

            #Most recent observations are up to step index
            recent_obs = self.storage.observations[:self.storage.step+1,:,:]
            recent_obs = torch.flip(recent_obs, [0])

            #Remaining observations are after step, with the bottom ones most recent
            later_obs = self.storage.observations[self.storage.step+1:,:,:]
            later_obs = torch.flip(later_obs, [0])

            #This tensor contains all observations in sequential order (top is most recent)
            reordered_observations = torch.cat((recent_obs, later_obs),dim=0)

            force_estimator_input = reordered_observations[:self.force_estimator.num_timesteps, :, :]

            #Force estimator expects input as num environments x num states x num features
            force_estimator_input = torch.transpose(force_estimator_input, 0, 1)

            estimated_force = self.force_estimator(force_estimator_input)

            obs = torch.cat((obs[:, :-6], estimated_base_vel),dim=-1)
            obs = torch.cat((obs, estimated_force),dim=-1)

        else:

            obs = torch.cat((obs[:, :-3], estimated_base_vel),dim=-1)


        # Compute the actions and values
        self.transition.actions = self.actor_critic.act(obs).detach()
        self.transition.values = self.actor_critic.evaluate(critic_obs).detach()
        self.transition.actions_log_prob = self.actor_critic.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.actor_critic.action_mean.detach()
        self.transition.action_sigma = self.actor_critic.action_std.detach()

        # need to record obs and critic_obs before env.step()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        return self.transition.actions
    
    def process_env_step(self, rewards, dones, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        # Bootstrapping on time outs
        if 'time_outs' in infos:
            self.transition.rewards += self.gamma * torch.squeeze(self.transition.values * infos['time_outs'].unsqueeze(1).to(self.device), 1)

        # Record the transition
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.actor_critic.reset(dones)
    
    def compute_returns(self, last_critic_obs):
        last_values= self.actor_critic.evaluate(last_critic_obs).detach()
        self.storage.compute_returns(last_values, self.gamma, self.lam)

    def update(self):

        #Update actor_critic AND state_estimator
        mean_value_loss = 0
        mean_surrogate_loss = 0
        if self.actor_critic.is_recurrent:
            generator = self.storage.reccurent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for obs_batch, critic_obs_batch, actions_batch, target_values_batch, advantages_batch, returns_batch, old_actions_log_prob_batch, \
            old_mu_batch, old_sigma_batch, hid_states_batch, masks_batch in generator:


                self.actor_critic.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
                actions_log_prob_batch = self.actor_critic.get_actions_log_prob(actions_batch)
                value_batch = self.actor_critic.evaluate(critic_obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
                mu_batch = self.actor_critic.action_mean
                sigma_batch = self.actor_critic.action_std
                entropy_batch = self.actor_critic.entropy

                # KL
                if self.desired_kl != None and self.schedule == 'adaptive':
                    with torch.inference_mode():
                        kl = torch.sum(
                            torch.log(sigma_batch / old_sigma_batch + 1.e-5) + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch)) / (2.0 * torch.square(sigma_batch)) - 0.5, axis=-1)
                        kl_mean = torch.mean(kl)

                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate


                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
                surrogate = -torch.squeeze(advantages_batch) * ratio
                surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(ratio, 1.0 - self.clip_param,
                                                                                1.0 + self.clip_param)
                surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

                # Value function loss
                if self.use_clipped_value_loss:
                    value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(-self.clip_param,
                                                                                                    self.clip_param)
                    value_losses = (value_batch - returns_batch).pow(2)
                    value_losses_clipped = (value_clipped - returns_batch).pow(2)
                    value_loss = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss = (returns_batch - value_batch).pow(2).mean()

                loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()


                #print(entropy_batch.mean())
                #print("Surrogate Loss:", surrogate_loss.item())
                #print("Value Loss:", value_loss.item())

                # Update actor_critic params
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

                # Update base vel estimator via supervised learning

                if(self.isForceEstimator):
                    true_base_vel = critic_obs_batch[:,-6:-3]
                else:
                    true_base_vel = critic_obs_batch[:,-3:]
                predicted_base_vel = self.base_velocity_estimator(obs_batch)
                base_velocity_estimator_computed_loss = self.base_velocity_estimator_loss(predicted_base_vel, true_base_vel)

                self.base_velocity_estimator_optimizer.zero_grad()
                base_velocity_estimator_computed_loss.backward()
                self.base_velocity_estimator_optimizer.step()

                #print("Base Vel Estimator Loss:", base_velocity_estimator_computed_loss.item())


                # # Update force estimator via supervised learning
                # true_force = critic_obs_batch[:,-3:]

                # #Only update force estimator if batch contains some non-zero forces
                # if(torch.count_nonzero(true_force).item() > 0):

                #     #Ensure only some percentage of batch is 0 forces
                #     percentage_zero_samples = 0.4
                #     num_nonzero_samples = int(torch.count_nonzero(true_force).item()/3)
                #     num_total_samples = int(num_nonzero_samples/(1-percentage_zero_samples))
                #     num_zero_samples =  num_total_samples - num_nonzero_samples

                #     zero_force_indicies = (true_force[:,0] == 0).nonzero()
                #     zero_force_indicies = zero_force_indicies[:num_zero_samples]

                #     non_zero_force_indicies = (true_force[:,0] != 0).nonzero()

                #     #Combine and shuffle indicies
                #     rebalanced_indicies = torch.cat(( zero_force_indicies, non_zero_force_indicies ),dim=0)
                #     rebalanced_indicies = rebalanced_indicies[torch.randperm(rebalanced_indicies.size()[0])]

                #     rebalanced_true_force = critic_obs_batch[rebalanced_indicies,-3:].squeeze(1)
                #     rebalanced_samples = obs_batch[rebalanced_indicies, :].squeeze(1)

                #     predicted_force = self.force_estimator(rebalanced_samples)

                #     force_estimator_computed_loss = self.force_estimator_loss(predicted_force, rebalanced_true_force)

                #     self.force_estimator_optimizer.zero_grad()
                #     force_estimator_computed_loss.backward()
                #     self.force_estimator_optimizer.step()

                #     print("Force Estimator Loss:", force_estimator_computed_loss.item())




        if(self.isForceEstimator):

            #Update force estimator directly from self.storage.observations and self.storage.privileged_observations
            #Force estimator input requires states in order, which batch does not maintain
            #At this point, self.storage.observations and self.storage.privileged_observations are already ordered.
            #most recent at bottom

            #We want most recent observations at top
            flipped_obs = torch.flip(self.storage.observations, [0])
            flipped_privileged_obs = torch.flip(self.storage.privileged_observations, [0])

            #Only train if at least some external forces appear in the past set of observations
            true_forces = flipped_privileged_obs[:, :, -3:]

            percent_zero_samples = 0.5
            num_states = self.force_estimator.num_timesteps

            if(torch.count_nonzero(true_forces).item() > 0):

                #Actor, Critic, and base velocity estimator also trained over 5 epochs
                for epoch in range(5):


                    #Can train over samples from [self.force_estimator.num_timesteps, self.storage.observations.shape[0]]
                    #This ensures each sample can go back far enough in history
                    #self.storage.observations is num states x num envs x state size
                    num_training_samples = flipped_obs.shape[0] - num_states

                    #Select random permutation of state indicies to train over
                    training_idx = torch.randperm(num_training_samples) + num_states

                    #Among all training indicies, ensure at most percent_zero_samples have 0 labels
                    true_forces = flipped_privileged_obs[training_idx, :, -3:]

                    num_nonzero_samples = int(torch.count_nonzero(true_forces).item()/3)
                    num_total_samples = int(num_nonzero_samples/(1-percent_zero_samples))
                    num_zero_samples =  num_total_samples - num_nonzero_samples


                    #Tensor of state x env indicies of zero samples
                    zero_force_indicies = (true_forces[:,:,0] == 0).nonzero()
                    #print("Total zero samples:", zero_force_indicies.shape[0])

                    #Select num_zero_samples number of indicies with 0 forces
                    #Still tensor of state x env indicies
                    selected_zero_force_indicies = zero_force_indicies[:num_zero_samples]
                    #print("Training over", selected_zero_force_indicies.shape[0], "zero samples")

                    #Select all indicies including non-zero forces
                    #Tensor of state x env indicies
                    selected_non_zero_force_indicies = (true_forces[:,:,0] != 0).nonzero()
                    rebalanced_training_indicies = torch.cat(( selected_zero_force_indicies, selected_non_zero_force_indicies),dim=0)


                    #Gather training dataset.
                    #Each X sample contains self.force_estimator.num_timesteps number of states
                        # num samples x num states x state length
                    #Each Y sample contains a single label

                    state_indicies = training_idx[rebalanced_training_indicies[:,0]]
                    env_indicies = rebalanced_training_indicies[:,1]

                    #We expect X to have shape num samples(46080) x num_states(25) x num features (48)
                    #num samples is so large, because each env can have up to 48-25 = 23 usable samples

                    X = torch.zeros(state_indicies.shape[0], num_states, flipped_obs.shape[2], device=self.device)

                    #We expect Y to have shape num samples(46080) x label(3)
                    Y = torch.zeros(state_indicies.shape[0], 3, device=self.device)

                    #Iterative solution. This is inneficient, but I can't figure out how to do this without a loop
                    for i in range(rebalanced_training_indicies.shape[0]):

                        sample = flipped_obs[state_indicies[i]-num_states:state_indicies[i], env_indicies[i], :]
                        X[i] = sample

                        label = flipped_privileged_obs[state_indicies[i], env_indicies[i], -3:]
                        Y[i] = label


                    #Its possible X can have 0 samples, when earlier states are the only ones with nonzero forces
                    if(X.shape[0] > 0):
                        
                        train_dataset = TensorDataset(X, Y)
                        train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)

                        avg_loss = 0

                        #train over batches from rebalanced training data
                        for data, label in train_dataloader:

                            predicted_force = self.force_estimator(data)

                            force_estimator_computed_loss = self.force_estimator_loss(predicted_force, label)

                            self.force_estimator_optimizer.zero_grad()
                            force_estimator_computed_loss.backward()
                            self.force_estimator_optimizer.step()

                            avg_loss += force_estimator_computed_loss.item()

                        print("Force Estimator Loss:", avg_loss/training_idx.shape[0])


        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss
