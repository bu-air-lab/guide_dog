
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

from mpc_controller import com_velocity_estimator
from mpc_controller import gait_generator as gait_generator_lib
from mpc_controller import locomotion_controller
from mpc_controller import openloop_gait_generator
from mpc_controller import raibert_swing_leg_controller
from mpc_controller import torque_stance_leg_controller

#uncomment the robot of choice
#from mpc_controller import laikago_sim as robot_sim
from mpc_controller import a1_sim as robot_sim


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

#With no forces, MPC reaches this point
MPC_TARGET = (0.4554415821253426, 0.0322272408571831)

def _run_example():
  """Runs the locomotion controller example."""
  


  for i in range(3):

    p = bullet_client.BulletClient(connection_mode=pybullet.GUI)    
           
    
    simulation_time_step = 0.001
    p.setTimeStep(simulation_time_step)
    p.setGravity(0, 0, -9.8)
    p.setAdditionalSearchPath(pd.getDataPath())
    
    p.loadURDF("plane.urdf")


    #p.resetBasePositionAndOrientation(ground_id,[0,0,0], [0,0,0,1])
    
    #p.changeDynamics(ground_id, -1, lateralFriction=1.0)
    
    robot_uid = p.loadURDF(robot_sim.URDF_NAME, robot_sim.START_POS)


    position, _ = p.getBasePositionAndOrientation(robot_uid)
    print("Init Position:", position)

    robot = robot_sim.SimpleRobot(p, robot_uid, simulation_time_step=simulation_time_step)
    
    #Order: FR, FL, RR, RL
    foot_joint_indicies = [5, 10, 15, 20]
    for joint_index in foot_joint_indicies:
        p.enableJointForceTorqueSensor(robot_uid, joint_index)

    controller = _setup_controller(robot)
    controller.reset()
    

    lin_speed, ang_speed = (0.5, 0., 0.), 0.
    _update_controller_params(controller, lin_speed, ang_speed)

    for i in range(200):

      # Needed before every call to get_action().
      controller.update()
      hybrid_action, info = controller.get_action()
      
      robot.Step(hybrid_action, 100, 10, 0, 0)

    position, _ = p.getBasePositionAndOrientation(robot_uid)
    print("Final Position:", position)

    p.disconnect()


def main(argv):
  del argv
  _run_example()


if __name__ == "__main__":
  app.run(main)


"""
Init Position: (0.012731, 0.002186, 0.320515)
Final Position: (0.4554415821253426, 0.0322272408571831, 0.28010765507632385)

Init Position: (0.012731, 0.002186, 0.320515)
Final Position: (0.4554415821253426, 0.0322272408571831, 0.28010765507632385)

Init Position: (0.012731, 0.002186, 0.320515)
Final Position: (0.4554415821253426, 0.0322272408571831, 0.28010765507632385)

"""