import time
import peakdetect

from bullet_env.bullet_env import BulletEnv
from bullet_env.blank_env import BlankEnv

from guide_dog_ppo.runners import OnPolicyRunner

import numpy as np
import torch

import pybullet as p
import matplotlib.pyplot as plt


def detect_force(estimated_forces):

    #Scale by 10, such that magnitude is similar to other force detector
    #Then, we can keep same params for peak detector
    #estimated_forces *= 10

    x_axis = [i for i in range(estimated_forces.shape[0])]

    #Detect LEFT/RIGHT forces
    #lr_peaks = peakdetect.peakdetect(estimated_forces, x_axis, lookahead=1, delta=0.25)
    lr_peaks = peakdetect.peakdetect(estimated_forces, x_axis, lookahead=1, delta=0.05)

    #Get indicies of LEFT and RIGHT peaks
    left_peaks = np.array(lr_peaks[0]).astype(int)
    if(left_peaks.shape[0] > 0):
        left_peaks = left_peaks[:,0]

    right_peaks = np.array(lr_peaks[1]).astype(int)
    if(right_peaks.shape[0] > 0):
        right_peaks = right_peaks[:,0]


    #We consider a detected force as a peak within the last 50 timesteps
    target_timestep = estimated_forces.shape[0] - 50 - 1
    for peak in left_peaks:
        if(peak >= target_timestep):
            return 'LEFT'

    for peak in right_peaks:
        if(peak >= target_timestep):
            return 'RIGHT'

    return 'NONE'



isForceDetector = True
train_vel_only = True

#Load env:
#env = BulletEnv(isGUI=False)
env = BulletEnv(isGUI=False, isForceDetector=isForceDetector)

#Load Policy
train_cfg_dict = {'algorithm': {'clip_param': 0.2, 'desired_kl': 0.01, 'entropy_coef': 0.01, 'gamma': 0.99, 'lam': 0.95, 'learning_rate': 0.001, 
                                'max_grad_norm': 1.0, 'num_learning_epochs': 5, 'num_mini_batches': 4, 'schedule': 'adaptive', 
                                'use_clipped_value_loss': True, 'value_loss_coef': 1.0}, 
                                'init_member_classes': {}, 
                                'policy': {'activation': 'elu', 'actor_hidden_dims': [512, 256, 128], 'critic_hidden_dims': [512, 256, 128], 'init_noise_std': 1.0}, 
                                'runner': {'algorithm_class_name': 'PPO', 'checkpoint': -1, 'experiment_name': 'flat_a1', 'load_run': -1, 'max_iterations': 500, 
                                'num_steps_per_env': 24, 'force_estimation_timesteps': 25, 'policy_class_name': 'ActorCritic', 'resume': True, 'resume_path': None, 'run_name': '', 'save_interval': 50}, 
                                'runner_class_name': 'OnPolicyRunner', 'seed': 1}

#ppo_runner = OnPolicyRunner(BlankEnv(), train_cfg_dict)
ppo_runner = OnPolicyRunner(BlankEnv(use_force_estimator=isForceDetector, train_vel_only=train_vel_only), train_cfg_dict)

#policy_name = "estimator4"
policy_name = "only_vel3"
ppo_runner.load("/home/david/Desktop/guide_dog/pybullet_val/saved_models/"+ policy_name + ".pt")

policy, base_vel_estimator, force_estimator = ppo_runner.get_inference_policy()

obs,_ = env.reset()

#Store past observations, most recent at top
#env x num states x obs length
obs_history = torch.zeros(1, force_estimator.num_timesteps, force_estimator.num_obs)#, device=self.device, dtype=torch.float)
vel_history = torch.zeros(1, force_estimator.num_timesteps, 6)#, device=self.device, dtype=torch.float)

estimated_forces = []

for env_step in range(400):

    #Shift all rows down 1 row (1 timestep)
    obs_history = torch.roll(obs_history, shifts=(0,1,0), dims=(0,1,0))
    vel_history = torch.roll(vel_history, shifts=(0,1,0), dims=(0,1,0))

    obs = torch.Tensor(obs)

    #Set most recent state as first
    obs_history[:,0,:] = obs

    with torch.no_grad():
        #Update obs with estimated base_vel (replace features at the end of obs)
        estimated_base_vel = base_vel_estimator(obs.unsqueeze(0))

    #Set most recent state as first
    # linear_vel = torch.Tensor(np.array(env.getBaseVel()))
    # vel_history[:,0,:3] = linear_vel
    vel_history[:,0,:3] = estimated_base_vel
    vel_history[:,0,3:] = obs[:3]


    with torch.no_grad():

        if(isForceDetector):
            if(not train_vel_only):
                estimated_force = force_estimator(obs_history)
            else:
                estimated_force = force_estimator(vel_history)
            estimated_forces.append(estimated_force.tolist()[0])


    if(isForceDetector):
        obs = torch.cat((obs[:-6], estimated_base_vel.squeeze(0)),dim=-1)
        obs = torch.cat((obs, estimated_force.squeeze(0)),dim=-1)
    else:
        obs = torch.cat((obs[:-3], estimated_base_vel.squeeze(0)),dim=-1)

    #print("Estimated Force Vector:", estimated_force)


    with torch.no_grad():
        action = policy(obs).detach()

    obs, rew, done, info = env.step(action.detach())


if(isForceDetector):

    estimated_forces = np.array(estimated_forces)[:,1]
    x_axis = [i for i in range(estimated_forces.shape[0])]

    detected_forces = []
    for t in range(100, 400, 25):

        forces = estimated_forces[:t]
        detected_force = detect_force(forces)
        if(detected_force != 'NONE'):
            print(t, detected_force)

    #plt.plot(t, estimated_forces[:,0])
    plt.plot(x_axis, estimated_forces)
    plt.savefig(policy_name + '.png')

