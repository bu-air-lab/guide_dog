import gym
from gym import spaces

import pybullet as p
import time
import pybullet_data as pd
import numpy as np
import torch

#FR, FL, RR, RL
#INIT_MOTOR_ANGLES = np.array([-0.1, 0.8, -1.5, 0.1, 0.8, -1.5, -0.1, 1.0, -1.5, 0.1, 1.0, -1.5])



#FL, FR, RL, RR
INIT_MOTOR_ANGLES = np.array([0.1, 0.8, -1.5, -0.1, 0.8, -1.5, 0.1, 1.0, -1.5, -0.1, 1.0, -1.5])
#INIT_MOTOR_ANGLES = np.array([0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8, 0, 0.9, -1.8])


#Torque Control
class BulletEnv(gym.Env):

    def __init__(self, isGUI=False, isForceDetector=True):

        self.num_privileged_obs = None
        self.num_obs = 120
        self.num_envs = 1
        self.num_actions = 12

        self.isGUI = isGUI
        self.isForceDetector = isForceDetector

        self._cam_dist = 1.0
        self._cam_yaw = 0
        self._cam_pitch = -30

        self.urdf_path = "a1/a1.urdf"

        if(self.isGUI):
            p.connect(p.GUI)
            p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])
        else:
            p.connect(p.DIRECT)


        p.setAdditionalSearchPath(pd.getDataPath())
        self.plane = p.loadURDF("plane.urdf")
        self.robot = p.loadURDF(self.urdf_path,[0,0,0.3])#, useFixedBase=1)
        p.setGravity(0,0,-9.8)

        self.time_step = 0.005

        p.setTimeStep(self.time_step)

        #Order: FR, FL, RR, RL
        self.foot_joint_indicies = [5, 10, 15, 20]

        #old ordering
        self.motor_ids = [1, 3, 4, 6, 8, 9, 11, 13, 14, 16, 18, 19]

        #new ordering
        self.reordered_motor_ids = [6, 8, 9, 1, 3, 4, 16, 18, 19, 11, 13, 14]        
        
        for joint_index in self.foot_joint_indicies:
            p.enableJointForceTorqueSensor(self.robot, joint_index)
     
        #Just needed for resetting
        self.init_position, self.init_orientation = p.getBasePositionAndOrientation(self.robot)

        #State/Action bounds all based on normalized values.
        #Not super accurate, however shouldn't have to be, for the purposes of deploying an already trained policy
        obs_space_lower_bound = np.array([-10 for i in range(45)])
        obs_space_upper_bound = np.array([10 for i in range(45)])

        #URDF
        #action_space_upper_bound = np.array([0.802851455917, 4.18879020479, -0.916297857297]*4)
        #action_space_lower_bound = np.array([-0.802851455917, -1.0471975512, -2.69653369433]*4)


        #Isaac
        action_space_upper_bound = np.array([0.7226, 3.9270, -1.0053]*4)
        action_space_lower_bound = np.array([-0.7226, -0.7854, -2.6075]*4)

        self.observation_space = spaces.Box(low=obs_space_lower_bound, high=obs_space_upper_bound, dtype=np.float32)
        self.action_space = spaces.Box(low=action_space_lower_bound, high=action_space_upper_bound, dtype=np.float32)

        #Initialize orientation variables
        self.current_joint_angles = INIT_MOTOR_ANGLES
        self.current_joint_velocities = [0 for i in range(12)]

        #Counts number of actions taken
        self.current_timestep = 0

        #number of environment actions taken until isDone=True
        self.max_timestep = 1000

        #Number of simulation steps we take per environment step
        self.action_repeat = 4 #50Hz

        self.torque_limits = [20, 55, 55]*4
        #self.torque_limits = [5, 10, 10]*4
        self.p_gains = 20
        self.d_gains = 0.5

        self.clip_action = 100

        self.past_dof_pos = []
        self.past_dof_vel = []

        for i in range(4):
            self.past_dof_pos.append(INIT_MOTOR_ANGLES)
            self.past_dof_vel.append([0 for x in range(12)])

        self.last_base_vel = [0, 0, 0]


    def compute_torques(self, action):

        scaled_action = [0.25*x.item() for x in action]

        target_pos = list(scaled_action) + INIT_MOTOR_ANGLES - self.current_joint_angles

        P = [self.p_gains*x for x in target_pos] 
        D = [self.d_gains*x for x in self.current_joint_velocities]
        torques = [P[i] - D[i] for i in range(12)]
        return np.clip(torques, [-x for x in self.torque_limits], self.torque_limits)


    def _StepInternal(self, action):

        torques = self.compute_torques(action)

        p.setJointMotorControlArray(
            bodyIndex=self.robot,
            jointIndices=self.reordered_motor_ids,
            controlMode=p.TORQUE_CONTROL,
            forces=torques)

        p.stepSimulation()

        if(self.isGUI): #Sleep for rendering only
            time.sleep(self.time_step*2)

            base_pos = p.getBasePositionAndOrientation(self.robot)[0]
            camInfo = p.getDebugVisualizerCamera()
            curTargetPos = camInfo[11]
            distance = camInfo[10]
            yaw = camInfo[8]
            pitch = camInfo[9]
            p.resetDebugVisualizerCamera(distance, yaw, pitch, base_pos)

        #Update joint angles and velocities
        joint_states = p.getJointStates(self.robot, self.reordered_motor_ids)
        joint_angles = [x[0] for x in joint_states]
        joint_velocities = [x[1] for x in joint_states]

        self.current_joint_angles = joint_angles
        self.current_joint_velocities = joint_velocities


    def step(self, action):

        isDone = False

        action = torch.clip(action, -self.clip_action, self.clip_action)

        #Take action_repeat number of simulation steps to complete action
        for i in range(self.action_repeat):
            self._StepInternal(action)

        self.current_timestep += 1

        self.state = self.getState(action)

        #Done if we reached the final timestep, or if we flip over
        if(self.current_timestep == self.max_timestep or not self.isHealthy()):
            isDone = True

        reward = 0
        info = {}

        return self.state, reward, isDone, info

    def reset(self):

        p.resetBasePositionAndOrientation(self.robot, self.init_position, self.init_orientation)
        p.resetBaseVelocity(self.robot, [0, 0, 0], [0, 0, 0])

        #Re-Initialize joint positions
        self.resetJoints()

        #Let robot start falling
        for i in range(4):
            p.stepSimulation()


        #Reset camera
        if(self.isGUI):
            p.resetDebugVisualizerCamera(self._cam_dist, self._cam_yaw, self._cam_pitch, [0, 0, 0])


        #self.current_joint_angles = pos
        #self.current_joint_velocities = vel
        self.current_joint_angles = INIT_MOTOR_ANGLES
        self.current_joint_velocities = [0 for i in range(12)]

        self.current_timestep = 0

        #initial action is to maintain default position??
        self.state = self.getState(torch.zeros(12))

        return self.state, None


    def getState(self, action):

        position, _ = p.getBasePositionAndOrientation(self.robot)
        print(position)

        #Update current joint angles and velocities
        joint_states = p.getJointStates(self.robot, self.reordered_motor_ids)
        joint_angles = [x[0] for x in joint_states]
        joint_velocities = [x[1] for x in joint_states]

        self.current_joint_angles = joint_angles
        self.current_joint_velocities = joint_velocities

        #Update joint angles and joint velocities history
        self.past_dof_pos.pop()
        self.past_dof_pos.insert(0, self.current_joint_angles.copy())

        self.past_dof_vel.pop()
        self.past_dof_vel.insert(0, self.current_joint_velocities.copy())

        #Compute state
        linear_vel, angular_vel = p.getBaseVelocity(self.robot)

        #print("True base vel:", [round(x,2) for x in linear_vel])
        #print([round(x,2) for x in self.current_joint_angles])

        applied_force = 000
        #if(((self.current_timestep % 50 == 0) or (self.current_timestep % 51 == 0) or (self.current_timestep % 52 == 0)) and self.current_timestep > 5):
        if(self.current_timestep % 100 == 0 and self.current_timestep > 5):

            print("APPLY FORCE")

            #Push right
            #p.applyExternalForce(objectUniqueId=self.robot, linkIndex=-1, forceObj=[0, -applied_force, 0], posObj=[0, 0, 0], flags=p.LINK_FRAME)

            #Push left
            p.applyExternalForce(objectUniqueId=self.robot, linkIndex=-1, forceObj=[0, applied_force, 0], posObj=[0, 0, 0], flags=p.LINK_FRAME)

            #Push back
            #p.applyExternalForce(objectUniqueId=self.robot, linkIndex=-1, forceObj=[-applied_force, 0, 0], posObj=[0, 0, 0], flags=p.LINK_FRAME)

            #Push forward
            #p.applyExternalForce(objectUniqueId=self.robot, linkIndex=-1, forceObj=[applied_force, 0, 0], posObj=[0, 0, 0], flags=p.LINK_FRAME)

        command = [1, 0, 0]
        #command = [0.2, 0, -0.4]

        acceleration = [(linear_vel[i] - self.last_base_vel[i])/ (self.time_step*self.action_repeat) for i in range(3)]
        self.last_base_vel = linear_vel

        #print("Acc:", [round(x,2) for x in acceleration])

        state = []
        #state.extend([2*x for x in linear_vel]) #Base velocity
        #state.extend([0.25*x for x in angular_vel]) #Angular Velocity
        state.extend(command) #Commands scale is (2, 2, 0.25). Command is [1, 0, 0]
        state.extend([x*0.25 for x in acceleration])
        state.extend(self.current_joint_angles - INIT_MOTOR_ANGLES) #Joint angles offset
        state.extend([x*0.05 for x in self.current_joint_velocities])  #Joint velocities
        state.extend(action.tolist())
        if(self.isForceDetector):
            state.extend([0,0,0,0,0,0])
        else:
            state.extend([0,0,0])

        return state    


    def resetJoints(self, reset_time=1.0):


        for index,_id in enumerate(self.reordered_motor_ids):

            #Reset joint angles
            p.resetJointState(self.robot, _id, INIT_MOTOR_ANGLES[index], targetVelocity=0)

            #Disable default motor to allow torque control later on
            p.setJointMotorControl2(self.robot, _id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

    #Must be balanced, only feet are allowed to touch ground, and body height must be above threshold
    def isHealthy(self):

        #Check balance
        isBalanced = True
        _, orientation = p.getBasePositionAndOrientation(self.robot)
        euler_orientation = p.getEulerFromQuaternion(orientation)
        if(euler_orientation[0] > 0.4 or euler_orientation[1] > 0.2):
            isBalanced = False

        #Check if anything except feet are in contact with ground
        isContact = False
        contact_points = p.getContactPoints(self.robot, self.plane)
        links = []
        for point in contact_points:
            robot_link = point[3]
            if(robot_link not in self.foot_joint_indicies):
                isContact = True

        isHealthy = (isBalanced and not isContact)

        return isHealthy
