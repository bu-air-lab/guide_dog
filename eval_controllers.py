
from __future__ import absolute_import
from __future__ import division
#from __future__ import google_type_annotations
from __future__ import print_function

import os
import inspect
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
os.sys.path.insert(0, parentdir)

from absl import app
from absl import flags
import scipy.interpolate
import numpy as np
import pybullet_data as pd
from pybullet_utils import bullet_client

import time
import pybullet
import random
import math

from mpc_controller import com_velocity_estimator
from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import locomotion_controller
from mpc_controller import openloop_gait_generator
from mpc_controller import raibert_swing_leg_controller
from mpc_controller import torque_stance_leg_controller

#uncomment the robot of choice
#from mpc_controller import laikago_sim as robot_sim
from mpc_controller import a1_sim as robot_sim


import time

from pybullet_val.bullet_env.eval_robustness_env import EvalRobustnessEnv
from pybullet_val.bullet_env.blank_env import BlankEnv

from guide_dog_ppo.runners import OnPolicyRunner

import numpy as np
import torch

import pybullet as p


_STANCE_DURATION_SECONDS = [
    0.15
] * 4   # reduce this number if we want higher velocity

# Trotting
_DUTY_FACTOR = [0.6] * 4
_INIT_PHASE_FULL_CYCLE = [0.9, 0, 0, 0.9]

_INIT_LEG_STATE = (
    gait_generator_lib.LegState.SWING,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.STANCE,
    gait_generator_lib.LegState.SWING,
)


def _setup_controller(robot):
  """Demonstrates how to create a locomotion controller."""
  desired_speed = (0, 0)
  desired_twisting_speed = 0

  gait_generator = openloop_gait_generator.OpenloopGaitGenerator(
      robot,
      stance_duration=_STANCE_DURATION_SECONDS,
      duty_factor=_DUTY_FACTOR,
      initial_leg_phase=_INIT_PHASE_FULL_CYCLE,
      initial_leg_state=_INIT_LEG_STATE)
  state_estimator = com_velocity_estimator.COMVelocityEstimator(robot,
                                                                window_size=20)
  sw_controller = raibert_swing_leg_controller.RaibertSwingLegController(
      robot,
      gait_generator,
      state_estimator,
      desired_speed=desired_speed,
      desired_twisting_speed=desired_twisting_speed,
      desired_height=robot_sim.MPC_BODY_HEIGHT,
      foot_clearance=0.01)

  st_controller = torque_stance_leg_controller.TorqueStanceLegController(
      robot,
      gait_generator,
      state_estimator,
      desired_speed=desired_speed,
      desired_twisting_speed=desired_twisting_speed,
      desired_body_height=robot_sim.MPC_BODY_HEIGHT,
      body_mass=robot_sim.MPC_BODY_MASS,
      body_inertia=robot_sim.MPC_BODY_INERTIA)

  controller = locomotion_controller.LocomotionController(
      robot=robot,
      gait_generator=gait_generator,
      state_estimator=state_estimator,
      swing_leg_controller=sw_controller,
      stance_leg_controller=st_controller,
      clock=robot.GetTimeSinceReset)
  return controller


def _update_controller_params(controller, lin_speed, ang_speed):
  controller.swing_leg_controller.desired_speed = lin_speed
  controller.swing_leg_controller.desired_twisting_speed = ang_speed
  controller.stance_leg_controller.desired_speed = lin_speed
  controller.stance_leg_controller.desired_twisting_speed = ang_speed


def is_healthy(robot_id, plane_id, p):

  #Check balance
  isBalanced = True
  position, orientation = p.getBasePositionAndOrientation(robot_id)
  euler_orientation = p.getEulerFromQuaternion(orientation)
  if(euler_orientation[0] > 0.4 or euler_orientation[1] > 0.2):
      isBalanced = False

  #Check if robot is high in air
  isTooHigh = False
  if(position[2] > 0.35):
    isTooHigh = True

  #isHealthy = (isBalanced and not isContact and not isTooHigh)
  isHealthy = (isBalanced and not isTooHigh)

  return isHealthy




#With no forces, MPC reaches this point
MPC_TARGET = (2.327454017235708, 0.060377823538094545)

def _run_mpc(force_vector, force_duration, force_start):

  #p = bullet_client.BulletClient(connection_mode=pybullet.GUI)    
  p = bullet_client.BulletClient(connection_mode=pybullet.DIRECT)

  simulation_time_step = 0.001
  p.setTimeStep(simulation_time_step)
  p.setGravity(0, 0, -9.8)
  p.setAdditionalSearchPath(pd.getDataPath())
  
  plane_id = p.loadURDF("plane.urdf")
  
  robot_uid = p.loadURDF(robot_sim.URDF_NAME, robot_sim.START_POS)

  robot = robot_sim.SimpleRobot(p, robot_uid, simulation_time_step=simulation_time_step)
  
  #Order: FR, FL, RR, RL
  foot_joint_indicies = [5, 10, 15, 20]
  for joint_index in foot_joint_indicies:
      p.enableJointForceTorqueSensor(robot_uid, joint_index)

  controller = _setup_controller(robot)
  controller.reset()
  

  lin_speed, ang_speed = (0.5, 0., 0.), 0.
  _update_controller_params(controller, lin_speed, ang_speed)

  isHealthy = True

  #1000 steps takes 0.001 * 5 * 1000 = 5 seconds
  for i in range(1000):

    # Needed before every call to get_action().
    controller.update()
    hybrid_action, info = controller.get_action()
    
    robot.Step(hybrid_action, force_vector, force_duration, force_start)

    if(not is_healthy(robot_uid, plane_id, p)):
      isHealthy = False
      break

  final_position, _ = p.getBasePositionAndOrientation(robot_uid)

  p.disconnect()

  distance_from_goal = math.sqrt(((final_position[0] - MPC_TARGET[0])**2) + (final_position[1] - MPC_TARGET[1])**2)

  return isHealthy, distance_from_goal

LEARNED_TARGETS_FORCE = [(2.4717457579273443, -0.13153568252551676),
                        (1.9886206438531304, 0.23275052617616304),
                        (2.3312287583138747, 0.17575623730110093),
                        (2.3524201953298554, 0.1955107740653754),
                        (2.3742899989820394, -0.11848172376926122)]

LEARNED_TARGETS_NO_FORCE = [(2.4640057399800988, -0.40527395382736786),
                            (2.1973504083564634, 0.4304406477621239),
                            (2.2487457915746356, -0.03530365722510473),
                            (2.358351938499667, 0.17532699777171643),
                            (2.1908176892470728, 0.24950895678853496)]

def _run_learned(force_vector, force_duration, force_start, policy_name, use_force_estimator):

  #Load Policy
  train_cfg_dict = {'algorithm': {'clip_param': 0.2, 'desired_kl': 0.01, 'entropy_coef': 0.01, 'gamma': 0.99, 'lam': 0.95, 'learning_rate': 0.001, 
                                  'max_grad_norm': 1.0, 'num_learning_epochs': 5, 'num_mini_batches': 4, 'schedule': 'adaptive', 
                                  'use_clipped_value_loss': True, 'value_loss_coef': 1.0}, 
                                  'init_member_classes': {}, 
                                  'policy': {'activation': 'elu', 'actor_hidden_dims': [512, 256, 128], 'critic_hidden_dims': [512, 256, 128], 'init_noise_std': 1.0}, 
                                  'runner': {'algorithm_class_name': 'PPO', 'checkpoint': -1, 'experiment_name': 'flat_a1', 'load_run': -1, 'max_iterations': 500, 
                                  'num_steps_per_env': 24, 'force_estimation_timesteps': 25, 'policy_class_name': 'ActorCritic', 'resume': True, 'resume_path': None, 'run_name': '', 'save_interval': 50}, 
                                  'runner_class_name': 'OnPolicyRunner', 'seed': 1}
  ppo_runner = OnPolicyRunner(BlankEnv(use_force_estimator=use_force_estimator), train_cfg_dict)
  ppo_runner.load("/home/david/Desktop/guide_dog/logs/guide_dog/" + policy_name + "/model_1500.pt")
  policy, base_vel_estimator, force_estimator = ppo_runner.get_inference_policy()


  obs_history = torch.zeros(1, force_estimator.num_timesteps, force_estimator.num_obs)#, device=self.device, dtype=torch.float)
  env = EvalRobustnessEnv(isGUI=False, isForceDetector=use_force_estimator)
  obs,_ = env.reset()

  isHealthy = True

  #250 steps takes 0.005 * 4 * 250 = 5 seconds
  for env_step in range(250):

    #print(env_step)

    #Shift all rows down 1 row (1 timestep)
    obs_history = torch.roll(obs_history, shifts=(0,1,0), dims=(0,1,0))
    obs = torch.Tensor(obs)

    #Set most recent state as first
    obs_history[:,0,:] = obs

    with torch.no_grad():
      #Update obs with estimated base_vel (replace features at the end of obs)
      estimated_base_vel = base_vel_estimator(obs.unsqueeze(0))

      if(use_force_estimator):
        estimated_force = force_estimator(obs_history)

    if(use_force_estimator):
      obs = torch.cat((obs[:-6], estimated_base_vel.squeeze(0)),dim=-1)
      obs = torch.cat((obs, estimated_force.squeeze(0)),dim=-1)
    else:
      obs = torch.cat((obs[:-3], estimated_base_vel.squeeze(0)),dim=-1)

    with torch.no_grad():
      action = policy(obs).detach()

    obs, rew, done, info = env.step(action.detach(), force_vector, force_duration, force_start)
    if(not is_healthy(env.robot, env.plane, env.get_bullet_client())):
      isHealthy = False
      break

  final_position, _ = env.get_bullet_client().getBasePositionAndOrientation(env.robot)

  env.close()

  seed_idx = int(policy_name[-1])-1

  if(use_force_estimator):
    distance_from_goal = math.sqrt(((final_position[0] - LEARNED_TARGETS_FORCE[seed_idx][0])**2) + (final_position[1] - LEARNED_TARGETS_FORCE[seed_idx][1])**2)
  else:
    distance_from_goal = math.sqrt(((final_position[0] - LEARNED_TARGETS_NO_FORCE[seed_idx][0])**2) + (final_position[1] - LEARNED_TARGETS_NO_FORCE[seed_idx][1])**2)

  return isHealthy, distance_from_goal


#Number of seconds per trial. 
#Must change MPC_TARGET and LEARNED_TARGET if this changes.
TRIAL_LENGTH = 5

#Define distributions to select force strength, duration, and starting point in episode
FORCE_RANGE = [25, 100] #in Newtons
FORCE_DURATION = [0.25, 0.5] #in seconds
FORCE_START = [1, 2] #in seconds

learned_isHealthy_lst = []
learned_isHealthy_noForce_lst = []
mpc_isHealthy_lst = []

learned_distances = []
learned_distances_noForce_lst = []
mpc_distances = []

num_trials = 1000

for trial in range(num_trials):

  print("Trial:", trial)

  #Select force
  force_magnitude = random.randint(FORCE_RANGE[0], FORCE_RANGE[1])
  force_duration = random.uniform(FORCE_DURATION[0], FORCE_DURATION[1])
  force_start = random.uniform(FORCE_START[0], FORCE_START[1])
  angle = random.uniform(0, 2*math.pi)

  x = math.cos(angle) * force_magnitude
  y = math.sin(angle) * force_magnitude
  force_vector = [x,y]

  mpc_isHealthy, mpc_distance_from_goal = _run_mpc(force_vector, force_duration, force_start)
  mpc_isHealthy_lst.append(mpc_isHealthy)
  mpc_distances.append(mpc_distance_from_goal)

  #Run trial for each seed of each learned policy
  for seed in range(1,6):

    #Without force estimator
    learned_isHealthy, learned_distance_from_goal = _run_learned(force_vector, force_duration, force_start, "no_estimator" + str(seed), False)
    learned_isHealthy_noForce_lst.append(learned_isHealthy)
    learned_distances_noForce_lst.append(learned_distance_from_goal)

    #With force estimator
    learned_isHealthy, learned_distance_from_goal = _run_learned(force_vector, force_duration, force_start, "with_estimator" + str(seed), True)
    learned_isHealthy_lst.append(learned_isHealthy)
    learned_distances.append(learned_distance_from_goal)

print("MPC isHealthy rate:", sum(mpc_isHealthy_lst)/num_trials)
print("Learned isHealthy No Force rate:", sum(learned_isHealthy_noForce_lst)/(num_trials*5))
print("Learned isHealthy With Force rate:", sum(learned_isHealthy_lst)/(num_trials*5))

print("MPC Avg distance:", sum(mpc_distances)/num_trials, np.std(mpc_distances))
print("Learned No Force Avg distance:", sum(learned_distances_noForce_lst)/(num_trials*5), np.std(learned_distances_noForce_lst))
print("Learned With Force Avg distance:", sum(learned_distances)/(num_trials*5), np.std(learned_distances))