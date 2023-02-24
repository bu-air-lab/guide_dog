import time

from bullet_env.bullet_env import BulletEnv
from bullet_env.blank_env import BlankEnv

from guide_dog_ppo.runners import OnPolicyRunner

import numpy as np
import torch

import pybullet as p

import pandas as pd

#Load env:
#env = BulletEnv(isGUI=False)
#env = BulletEnv(isGUI=True)

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
#ppo_runner.load("/home/dave/Desktop/guide_dog/pybullet_val/saved_models/baseline.pt")
ppo_runner.load("/home/eisuke/WORKSPACE/isaac/guide_dog/pybullet_val/saved_models/v29.pt")

policy, base_vel_estimator, force_estimator = ppo_runner.get_inference_policy()

#d = {'x_force':[0], 'y_force':[0], 'est_x_force':[0], 'est_y_force':[0], 'start':[0], 'duration':[0], 'healthy':[False], 'step':[0]} # fill in values for the 0th row
#d = {'x_force':[0], 'y_force':[0], 'start':[0], 'duration':[0], 'healthy':[False], 'step':[0]} # fill in values for the 0th row
d = {'x_force':[0], 'y_force':[0], 'start':[0], 'duration':[0], 'pose_x':[0], 'pose_y':[0], 'healthy':[False], 'step':[0]} # fill in values for the 0th row
# d = {'pose_x':[0], 'pose_y':[0]} # fill in values for the 0th row

for trial in range(100):
    #Store past observations, most recent at top
    #env x num states x obs length
    obs_history = torch.zeros(1, force_estimator.num_timesteps, force_estimator.num_obs)#, device=self.device, dtype=torch.float)
    print(f"Trial {trial}")
    #env = BulletEnv(isGUI=True)
    env = BulletEnv(isGUI=False)
    obs,_ = env.reset()
    gt_force, start, duration = env.get_test_info()

    # pose data # this pose is with command [1,0,0] or velocity 0.5m/s in x direction only
    # pose_x, pose_y, _ = env.get_robot_pose()
    # d['pose_x'].append(pose_x)
    # d['pose_y'].append(pose_y)

    # force data
    d['x_force'].append(gt_force[0])
    d['y_force'].append(gt_force[1])
    d['start'].append(start)
    d['duration'].append(duration)
    t1 = time.time()
    for env_step in range(1001):

        #Shift all rows down 1 row (1 timestep)
        obs_history = torch.roll(obs_history, shifts=(0,1,0), dims=(0,1,0))
        obs = torch.Tensor(obs)
        #Set most recent state as first
        obs_history[:,0,:] = obs

        with torch.no_grad():
            #Update obs with estimated base_vel (replace features at the end of obs)
            estimated_base_vel = base_vel_estimator(obs.unsqueeze(0))
            estimated_force = force_estimator(obs_history)

        obs = torch.cat((obs[:-6], estimated_base_vel.squeeze(0)),dim=-1)
        obs = torch.cat((obs, estimated_force.squeeze(0)),dim=-1)

        with torch.no_grad():
            action = policy(obs).detach()

        obs, rew, done, info = env.step(action.detach()) # this takes like 0.0005 seconds
        # pose_x, pose_y, _ = env.get_robot_pose()
        # d['pose_x'].append(pose_x)
        # d['pose_y'].append(pose_y)

        if done:
            healthy = env.isHealthy()
            d['healthy'].append(healthy)
            d['step'].append(info['step'])
            pose_x, pose_y, _ = env.get_robot_pose()
            d['pose_x'].append(pose_x)
            d['pose_y'].append(pose_y)

            #d['est_x_force'].append(estimated_force[0][0].cpu().item())
            #d['est_y_force'].append(estimated_force[0][1].cpu().item())
            break

    env.close()
    t2 = time.time()
    print(t2 - t1) # ~15 seconds in GUI
print("finished data collection, converting data to df")
df = pd.DataFrame(d)
print("converted to df. starting saving as csv")
#df.to_csv("test_policy_1.csv")
#df.to_csv("test_policy_2.csv")
df.to_csv("test_policy_3.csv")
#df.to_csv("policy_pose.csv")
print("Done")