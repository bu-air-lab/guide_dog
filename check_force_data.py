import pandas as pd
import numpy as np
import math

# mpc = pd.read_csv('test_mpc_2.csv')
mpc = pd.read_csv('test_mpc_3.csv')
#policy = pd.read_csv('test_policy_2.csv')
policy = pd.read_csv('test_policy_3.csv')

# shared data
avg_start = policy['start'][1:].mean() # start from index 1 bc index 0 is throwaway
avg_duration = policy['duration'][1:].mean()
avg_x_forces = policy['x_force'][1:].to_numpy().mean()
avg_y_forces = policy['y_force'][1:].to_numpy().mean()

# init mpc data
mpc_healthy_count = 0
mpc_step_sum = 0
mpc_healthy = (mpc['healthy'][1:].to_numpy()*1).mean()
mpc_step = (mpc['step'][1:].to_numpy()).mean()/4
# mpc_x_pose = (mpc['pose_x'][1:].to_numpy()).mean() - 1.8609396638954112
# mpc_y_pose = (mpc['pose_y'][1:].to_numpy()).mean() - 0.05396660585693605
mpc_all_x_pose = (mpc['healthy'][1:].to_numpy()*1)* (mpc['pose_x'][1:].to_numpy())
mpc_all_y_pose = (mpc['healthy'][1:].to_numpy()*1)* (mpc['pose_y'][1:].to_numpy())
mpc_x_pose = (mpc_all_x_pose[mpc_all_x_pose != 0]).mean() - 1.8609396638954112
mpc_y_pose = (mpc_all_y_pose[mpc_all_y_pose != 0]).mean()- 0.05396660585693605

# init policy data
policy_healthy_count = 0
policy_step_sum = 0
policy_healthy = (policy['healthy'][1:].to_numpy()*1).mean()
policy_step = (policy['step'][1:].to_numpy()).mean()
# policy_x_pose = (policy['pose_x'][1:].to_numpy()).mean() - 1.6731545320355745
# policy_y_pose = (policy['pose_y'][1:].to_numpy()).mean() + 0.07990217919560628 # this num is negative hence plus
policy_all_x_pose = (policy['healthy'][1:].to_numpy()*1)* (policy['pose_x'][1:].to_numpy())
policy_all_y_pose = (policy['healthy'][1:].to_numpy()*1)* (policy['pose_y'][1:].to_numpy())
policy_x_pose = (policy_all_x_pose[policy_all_x_pose != 0]).mean() - 1.8609396638954112
policy_y_pose = (policy_all_y_pose[policy_all_y_pose != 0]).mean()- 0.05396660585693605

print("***************************")
print(f"Average start timestep: {avg_start}, Average force timestep duration: {avg_duration}")
print(f"Average force applied: {avg_x_forces, avg_y_forces}")
print("***************************")
print("MPC Data:")
print(f"Success rate: {mpc_healthy}")
print(f"Average trial length: {mpc_step}")
print(f"Average distance: {mpc_x_pose, mpc_y_pose}, {math.sqrt(mpc_x_pose**2 + mpc_y_pose**2)}")
print("***************************")
print("Policy Data:")
print(f"Success rate: {policy_healthy}")
print(f"Average trial length: {policy_step}")
print(f"Average distance: {policy_x_pose, policy_y_pose}, {math.sqrt(policy_x_pose**2 + policy_y_pose**2)}")