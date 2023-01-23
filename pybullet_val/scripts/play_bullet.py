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
                                'num_steps_per_env': 24, 'policy_class_name': 'ActorCritic', 'resume': True, 'resume_path': None, 'run_name': '', 'save_interval': 50}, 
                                'runner_class_name': 'OnPolicyRunner', 'seed': 1}
ppo_runner = OnPolicyRunner(BlankEnv(), train_cfg_dict)
#ppo_runner.load("/home/dave/Desktop/guide_dog/pybullet_val/saved_models/baseline.pt")
ppo_runner.load("/home/david/Desktop/guide_dog/pybullet_val/saved_models/v19.pt")

policy, state_estimator = ppo_runner.get_inference_policy()

obs,_ = env.reset()

#First command is all 0's
#obs = [0 for x in range(42)]

#obs = torch.Tensor(obs).unsqueeze(0)

for env_step in range(1000):

    obs = torch.Tensor(obs)#.unsqueeze(0)


    #Update obs with estimated base_vel (replace features at the end of obs)
    estimated_state = state_estimator(obs.unsqueeze(0))
    obs = torch.cat((obs[:-6], estimated_state[0]),dim=-1)

    #print("Estimated Base Velocity:", [round((x/2),2) for x in estimated_state[0][:3].tolist()])
    print("Estimated Force:", [round((x/2),2) for x in estimated_state[0][3:].tolist()])

    linear_vel, _ = p.getBaseVelocity(env.robot)

    #print("True base vel:", [round(x,2) for x in linear_vel])

    #print("Error:", np.abs((estimated_state[0][0].item()/2) - linear_vel[0]))

    action = policy(torch.Tensor(obs)).detach()

    #print("Action:", action)

    obs, rew, done, info = env.step(action.detach())

