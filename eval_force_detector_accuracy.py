"""
Run many trials, with each trial applying either a left or right force, sampled from some distribution
Measure how accurately our force detector picks up these forces
"""

import torch
import numpy as np
import random
import peakdetect

from guide_dog_ppo.runners import OnPolicyRunner

from pybullet_val.bullet_env.eval_robustness_env import EvalRobustnessEnv
from pybullet_val.bullet_env.blank_env import BlankEnv

#Detect forces via peak detection on estimated_forces vectors
#Implemented via peakdetect library: https://github.com/avhn/peakdetect
def detect_force(estimated_forces):

    x_axis = [i for i in range(estimated_forces.shape[0])]

    #Detect LEFT/RIGHT forces
    lr_peaks = peakdetect.peakdetect(estimated_forces, x_axis, lookahead=1, delta=0.25)

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


#Load Policy
train_cfg_dict = {'algorithm': {'clip_param': 0.2, 'desired_kl': 0.01, 'entropy_coef': 0.01, 'gamma': 0.99, 'lam': 0.95, 'learning_rate': 0.001, 
                              'max_grad_norm': 1.0, 'num_learning_epochs': 5, 'num_mini_batches': 4, 'schedule': 'adaptive', 
                              'use_clipped_value_loss': True, 'value_loss_coef': 1.0}, 
                              'init_member_classes': {}, 
                              'policy': {'activation': 'elu', 'actor_hidden_dims': [512, 256, 128], 'critic_hidden_dims': [512, 256, 128], 'init_noise_std': 1.0}, 
                              'runner': {'algorithm_class_name': 'PPO', 'checkpoint': -1, 'experiment_name': 'flat_a1', 'load_run': -1, 'max_iterations': 500, 
                              'num_steps_per_env': 24, 'force_estimation_timesteps': 25, 'policy_class_name': 'ActorCritic', 'resume': True, 'resume_path': None, 'run_name': '', 'save_interval': 50}, 
                              'runner_class_name': 'OnPolicyRunner', 'seed': 1}
ppo_runner = OnPolicyRunner(BlankEnv(use_force_estimator=True), train_cfg_dict)

accuracies = []
false_positive_percentages = []

#Evaluate on 5 different policies trained on different random seeds
for seed in range(1,6):

	ppo_runner.load("/home/david/Desktop/guide_dog/logs/guide_dog/estimator" + str(seed) + "/model_1500.pt")
	policy, base_vel_estimator, force_estimator = ppo_runner.get_inference_policy()

	num_trials = 1000
	num_correct = 0
	percent_false_positives_lst = []

	#For each policy, try many different forces to evaluate force detection accuracy
	for trial in range(num_trials):

		obs_history = torch.zeros(1, force_estimator.num_timesteps, force_estimator.num_obs)#, device=self.device, dtype=torch.float)
		env = EvalRobustnessEnv(isGUI=False, isForceDetector=True)
		obs,_ = env.reset()

		#Sample force vector (only push in y-direction. we only want left or right pushes) and force duration
		force_strength = random.randint(40, 100)
		bit = random.randint(0,1)
		if(bit == 0):
			force_strength = -force_strength
		force_vector = [0, force_strength, 0]
		force_duration = random.uniform(0.25, 0.5)
		force_start = random.uniform(2, 3)

		#Keep track of all estimated forces
		estimated_forces = []

		#Run policy at 0.5 m/s in PyBullet, apply left and right force sampled from some distribution
		#250 steps takes 0.005 * 4 * 250 = 5 seconds
		for env_step in range(250):

			#Shift all rows down 1 row (1 timestep)
			obs_history = torch.roll(obs_history, shifts=(0,1,0), dims=(0,1,0))
			obs = torch.Tensor(obs)

			#Set most recent state as first
			obs_history[:,0,:] = obs

			with torch.no_grad():
				#Update obs with estimated base_vel and force (replace features at the end of obs)
				estimated_base_vel = base_vel_estimator(obs.unsqueeze(0))
				estimated_force = force_estimator(obs_history)
				estimated_forces.append(estimated_force.tolist()[0])

			obs = torch.cat((obs[:-6], estimated_base_vel.squeeze(0)),dim=-1)
			obs = torch.cat((obs, estimated_force.squeeze(0)),dim=-1)

			with torch.no_grad():
				action = policy(obs).detach()

			obs, rew, done, info = env.step(action.detach(), force_vector, force_duration, force_start)

		env.close()


		#Iterate through signal in chunks of 25 timesteps, detecting signals in the past 50 timesteps
		#Ignore the first 100 timesteps, they tend to contain more noise
		estimated_forces = np.array(estimated_forces)[:,1]
		detected_forces = []
		for t in range(100, 250, 25):

		    forces = estimated_forces[:t]
		    detected_forces.append(detect_force(forces))

		#Determine whether the true force was predicted
		isCorrect = False
		if(force_vector[1] > 0 and 'LEFT' in detected_forces):
			isCorrect = True
		elif(force_vector[1] <= 0 and 'RIGHT' in detected_forces):
			isCorrect = True

		if(isCorrect):
			num_correct += 1

		#Determine percent of timestesps there is a false positve
		num_false_positives = detected_forces.count('LEFT') + detected_forces.count('RIGHT')
		if(isCorrect):
			num_false_positives -= 1

		percent_false_positives = num_false_positives/len(detected_forces)
		percent_false_positives_lst.append(percent_false_positives)

	print("Accuracy:", num_correct/num_trials)
	print("Avg False Positive Ratio:", np.mean(percent_false_positives_lst))

	accuracies.append(num_correct/num_trials)
	false_positive_percentages.append(np.mean(percent_false_positives_lst))

print("Avg Accuracy:", np.mean(accuracies))
print("Accuracy STD:", np.std(accuracies))
print("False positives:", np.mean(false_positive_percentages))
print("False positive STD:", np.std(false_positive_percentages))