from __future__ import absolute_import
from __future__ import division
#from __future__ import google_type_annotations
from __future__ import print_function

import time

from bullet_env.bullet_env import BulletEnv
from bullet_env.blank_env import BlankEnv

import numpy as np
import torch

import pybullet as p
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
import pandas as pd

# from scripts import com_velocity_estimator
# from scripts import gait_generator as gait_generator_lib
# from scripts import locomotion_controller
# from scripts import openloop_gait_generator
# from scripts import raibert_swing_leg_controller
# from scripts import torque_stance_leg_controller
# from scripts import a1_sim as robot_sim

from mpc_controller import com_velocity_estimator
from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import locomotion_controller
from mpc_controller import openloop_gait_generator
from mpc_controller import raibert_swing_leg_controller
from mpc_controller import torque_stance_leg_controller
from mpc_controller import a1_sim as robot_sim

FLAGS = flags.FLAGS

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

#Load env:
#env = BulletEnv(isGUI=True)
#env = BulletEnv(isGUI=False)

# robot = robot_sim.SimpleRobot(p, robot_uid, simulation_time_step=0.001)

# d = {'x_force':[0], 'y_force':[0], 'start':[0], 'duration':[0], 'healthy':[False], 'step':[0]} # fill in values for the 0th row
d = {'x_force':[0], 'y_force':[0], 'start':[0], 'duration':[0], 'pose_x':[0], 'pose_y':[0], 'healthy':[False], 'step':[0]} # fill in values for the 0th row
# d = {'pose_x':[0], 'pose_y':[0]} # fill in values for the 0th row


for trial in range(10):

    env = BulletEnv(isGUI=True)
    robot_uid = env.robot_info()
    env.reset()
    robot = robot_sim.SimpleRobot(p, robot_uid, simulation_time_step=0.001)
    controller = _setup_controller(robot)
    controller.reset()
    print("Trial num:", trial)
    gt_force, start, duration = env.get_test_info()

    # pose data
    # pose_x, pose_y, _ = env.get_robot_pose()
    # d['pose_x'].append(pose_x)
    # d['pose_y'].append(pose_y)

    # force data
    d['x_force'].append(gt_force[0])
    d['y_force'].append(gt_force[1])
    d['start'].append(start*4)
    d['duration'].append(duration*4)

    t1 = time.time()
    for env_step in range(2001):

        lin_speed, ang_speed = (0.5, 0., 0.), 0.
        _update_controller_params(controller, lin_speed, ang_speed)

        # Needed before every call to get_action().
        controller.update()
        hybrid_action, info = controller.get_action() # this action takes 0.001 - 0.01 seconds while others are 1e-5

        done, step = robot.Step(hybrid_action, start, duration, gt_force[0], gt_force[1]) # this action takes like 0.006 consistently

        # pose_x, pose_y, _ = env.get_robot_pose()
        # d['pose_x'].append(pose_x)
        # d['pose_y'].append(pose_y)
        if done:
            healthy = env.isHealthy()
            d['healthy'].append(healthy)
            d['step'].append(step)
            pose_x, pose_y, _ = env.get_robot_pose()
            d['pose_x'].append(pose_x)
            d['pose_y'].append(pose_y)
            break
    env.close()
    t2 = time.time()
    print(t2 - t1) # ~1.0 secs
print("finished data collection, converting data to df")
df = pd.DataFrame(d)
print("converted to df. starting saving as csv")
# df.to_csv("test_mpc_1.csv")
#df.to_csv("test_mpc_2.csv")
df.to_csv("test_mpc_3.csv")
#df.to_csv("mpc_pose.csv")
print("Done")