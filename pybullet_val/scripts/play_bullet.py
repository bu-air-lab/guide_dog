import time

from bullet_env.bullet_env import BulletEnv
from bullet_env.blank_env import BlankEnv

from guide_dog_ppo.runners import OnPolicyRunner

import numpy as np
import torch

import pybullet as p

#Load env:
#env = BulletEnv(isGUI=False)
env = BulletEnv(isGUI=True)

#Load Policy
train_cfg_dict = {'algorithm': {'clip_param': 0.2, 'desired_kl': 0.01, 'entropy_coef': 0.01, 'gamma': 0.99, 'lam': 0.95, 'learning_rate': 0.001, 
                                'max_grad_norm': 1.0, 'num_learning_epochs': 5, 'num_mini_batches': 4, 'schedule': 'adaptive', 
                                'use_clipped_value_loss': True, 'value_loss_coef': 1.0}, 
                                'init_member_classes': {}, 
                                'policy': {'activation': 'elu', 'actor_hidden_dims': [512, 256, 128], 'critic_hidden_dims': [512, 256, 128], 'init_noise_std': 1.0}, 
                                'runner': {'algorithm_class_name': 'PPO', 'checkpoint': -1, 'experiment_name': 'flat_a1', 'load_run': -1, 'max_iterations': 500, 
                                'num_steps_per_env': 24, 'force_estimation_timesteps': 25, 'policy_class_name': 'ActorCritic', 'resume': True, 'resume_path': None, 'run_name': '', 'save_interval': 50}, 
                                'runner_class_name': 'OnPolicyRunner', 'seed': 1}
ppo_runner = OnPolicyRunner(BlankEnv(), train_cfg_dict)

ppo_runner.load("/home/david/Desktop/guide_dog/pybullet_val/saved_models/v29.pt")

#v32_x all good

policy, base_vel_estimator, force_estimator = ppo_runner.get_inference_policy()

obs,_ = env.reset()

#Store past observations, most recent at top
#env x num states x obs length
obs_history = torch.zeros(1, force_estimator.num_timesteps, force_estimator.num_obs)#, device=self.device, dtype=torch.float)

for env_step in range(1000):

    #Shift all rows down 1 row (1 timestep)
    obs_history = torch.roll(obs_history, shifts=(0,1,0), dims=(0,1,0))

    obs = torch.Tensor(obs)

    #Set most recent state as first
    obs_history[:,0,:] = obs

    #print(obs_history[:,0:3,3:40])


    with torch.no_grad():

        #Update obs with estimated base_vel (replace features at the end of obs)
        #start_time = time.time()
        estimated_base_vel = base_vel_estimator(obs.unsqueeze(0))
        #print("Vel estimation:", (time.time() - start_time))

        #start_time = time.time()
        estimated_force = force_estimator(obs_history)
        #print("force estimation:", (time.time() - start_time))


    obs = torch.cat((obs[:-6], estimated_base_vel.squeeze(0)),dim=-1)
    obs = torch.cat((obs, estimated_force.squeeze(0)),dim=-1)

    #0.0285, -0.0095,
    #print(estimated_force[0][:2])

    #Subtract de-scaled velocity command from estimated force
    #vel_command = obs[:3]
    #estimated_force_vector = [estimated_force[0][i] + vel_command[i]/2 for i in range(2)]

    #Add Bias to center force estimates around 0 on no forces
    #X: -0.02 -> -0.1
    #Y: 0.01 -> -0.14
    #bias = [0.07, 0.06]
    bias = [-0.47, 0.05]
    #estimated_force_vector = [round((estimated_force_vector[i] + bias[i]).item(),2) for i in range(2)]
    estimated_force_vector = [round((estimated_force[0][i] + bias[i]).item(),2) for i in range(2)]

    print("Estimated Force Vector:", estimated_force_vector)

    #linear_vel, _ = p.getBaseVelocity(env.robot)

    with torch.no_grad():

        #start_time = time.time()
        action = policy(obs).detach()
        #print("Policy Query:", (time.time() - start_time))

    obs, rew, done, info = env.step(action.detach())
