import torch
import torch.nn as nn
import torch.optim as optim

from guide_dog_ppo.modules import ActorCritic
from guide_dog_ppo.storage import RolloutStorage

class PPO:
    actor_critic: ActorCritic
    def __init__(self,
                 actor_critic,
                 base_velocity_estimator,
                 force_estimator,
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
        estimated_force = self.force_estimator(obs)

        obs = torch.cat((obs[:, :-6], estimated_base_vel),dim=-1)
        obs = torch.cat((obs, estimated_force),dim=-1)

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

                # Update actor_critic params
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

                # Update base vel estimator via supervised learning
                true_base_vel = critic_obs_batch[:,-6:-3]
                predicted_base_vel = self.base_velocity_estimator(obs_batch)

                base_velocity_estimator_computed_loss = self.base_velocity_estimator_loss(predicted_base_vel, true_base_vel)

                self.base_velocity_estimator_optimizer.zero_grad()
                base_velocity_estimator_computed_loss.backward()
                self.base_velocity_estimator_optimizer.step()

                #print("Base Vel Estimator Loss:", base_velocity_estimator_computed_loss.item())

                # Update force estimator via supervised learning
                true_force = critic_obs_batch[:,-3:]

                #Only update force estimator if batch contains some non-zero forces
                if(torch.count_nonzero(true_force).item() > 0):

                    #Ensure only some percentage of batch is 0 forces
                    percentage_zero_samples = 0.0001
                    num_nonzero_samples = int(torch.count_nonzero(true_force).item()/3)
                    num_total_samples = int(num_nonzero_samples/(1-percentage_zero_samples))
                    num_zero_samples =  num_total_samples - num_nonzero_samples


                    zero_force_indicies = (true_force[:,0] == 0).nonzero()
                    selected_zero_samples = critic_obs_batch[zero_force_indicies[:num_zero_samples]]

                    non_zero_force_indicies = (true_force[:,0] != 0).nonzero()
                    selected_nonzero_samples = critic_obs_batch[non_zero_force_indicies]

                    #Re-combine selected samples and randomly shuffle
                    rebalanced_critic_obs_batch = torch.cat(( selected_nonzero_samples, selected_zero_samples ),dim=0)
                    rebalanced_critic_obs_batch = rebalanced_critic_obs_batch[torch.randperm(rebalanced_critic_obs_batch.size()[0])]
                    rebalanced_critic_obs_batch = rebalanced_critic_obs_batch.squeeze(1)

                    rebalanced_true_force = rebalanced_critic_obs_batch[:,-3:]

                    #Add estimated base velocities to observations
                    #predicted_base_vel = self.base_velocity_estimator(rebalanced_critic_obs_batch)
                    #rebalanced_critic_obs_batch[:,-6:-3] = predicted_base_vel.detach()

                    #Predict applied force
                    predicted_force = self.force_estimator(rebalanced_critic_obs_batch)

                    #print(predicted_force, rebalanced_true_force)
                    #print("------------------------")

                    force_estimator_computed_loss = self.force_estimator_loss(predicted_force, rebalanced_true_force)

                    self.force_estimator_optimizer.zero_grad()
                    force_estimator_computed_loss.backward()
                    self.force_estimator_optimizer.step()

                    print("Force Estimator Loss:", force_estimator_computed_loss.item())


        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        self.storage.clear()

        return mean_value_loss, mean_surrogate_loss
