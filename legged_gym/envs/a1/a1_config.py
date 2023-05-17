# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

class GuideDogCfg( LeggedRobotCfg ):
    class init_state( LeggedRobotCfg.init_state ):
        pos = [0.0, 0.0, 0.42] # x,y,z [m]
        default_joint_angles = { # = target angles [rad] when action = 0.0
            'FL_hip_joint': 0.1,   # [rad]
            'RL_hip_joint': 0.1,   # [rad]
            'FR_hip_joint': -0.1 ,  # [rad]
            'RR_hip_joint': -0.1,   # [rad]

            'FL_thigh_joint': 0.8,     # [rad]
            'RL_thigh_joint': 1.,   # [rad]
            'FR_thigh_joint': 0.8,     # [rad]
            'RR_thigh_joint': 1.,   # [rad]

            'FL_calf_joint': -1.5,   # [rad]
            'RL_calf_joint': -1.5,    # [rad]
            'FR_calf_joint': -1.5,  # [rad]
            'RR_calf_joint': -1.5,    # [rad]
        }

    class env ( LeggedRobotCfg.env ):

        num_envs = 2048 #4096
        num_observations = 48 #120
        num_privileged_obs = 48 #120

        min_base_height = 0.25

        use_force_estimator = True

        isRAO = False
        rao_torque_range = [-5, 5]


    class commands:
        class ranges:
            lin_vel_x = [-1, 1] # min max [m/s]
            lin_vel_y = [-1, 1]   # min max [m/s]
            ang_vel_yaw = [-1, 1]    # min max [rad/s]
            heading = [-3.14, 3.14]


    class domain_rand ( LeggedRobotCfg.domain_rand ):

        randomize_base_mass = False

        push_robots = True
        push_interval_s = 3 #15 #How often to push (lower means more frequent)
        max_push_vel = 0.75 #1 #Max push velocity in xy directions
        max_z_vel = 0.1 #Max push velocity in z direction
        push_length_interval = [12, 24]

        back_push_interval_s = 0.6 #Simulate human holding rigid handle connected to robot
        back_push_length = 5
        back_push_vel = 0.25

        randomize_friction = True

    class terrain( LeggedRobotCfg.terrain ):

        #Only train over random_uniform_noise terrain
        terrain_proportions = [0, 1.0, 0, 0, 0]
        #mesh_type = 'plane'


    class commands( LeggedRobotCfg.commands ):

        heading_command = False #Directly sample angular velocity command

    class control( LeggedRobotCfg.control ):
        # PD Drive parameters:
        control_type = 'P'
        stiffness = {'joint': 20.}  # [N*m/rad]
        damping = {'joint': 0.5}     # [N*m*s/rad]
        # action scale: target angle = actionScale * action + defaultAngle
        action_scale = 0.25
        # decimation: Number of control action updates @ sim DT per policy DT
        decimation = 4

    class noise ( LeggedRobotCfg.noise ):

        add_noise = True

        class noise_scales ( LeggedRobotCfg.noise.noise_scales ):
            dof_pos = 0.01
            dof_vel = 1.5
            lin_vel = 0.1
            ang_vel = 0.2
            gravity = 0.05
            base_acc = 1.0
            base_orientation = 0.05
            height_measurements = 0.1

    class normalization ( LeggedRobotCfg.normalization ):
        class obs_scales ( LeggedRobotCfg.normalization.obs_scales ):
            lin_vel = 2.0
            ang_vel = 0.25
            dof_pos = 1.0
            dof_vel = 0.05
            base_acc = 0.25
            height_measurements = 5.0
        clip_observations = 100.
        clip_actions = 100.
  
    class rewards( LeggedRobotCfg.rewards ):
        soft_dof_pos_limit = 0.9
        base_height_target = 0.25
        tracking_sigma = 0.25 # tracking reward = exp(-error^2/sigma)

        class scales( LeggedRobotCfg.rewards.scales ):
            torques = -0.0002
            dof_pos_limits = -10.0

    class asset( LeggedRobotCfg.asset ):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/a1/urdf/a1.urdf'
        name = "a1"
        foot_name = "foot"
        #penalize_contacts_on = ["thigh", "calf"]
        #terminate_after_contacts_on = ["base"]
        terminate_after_contacts_on = ["thigh", "calf", "base"]
        self_collisions = 1 # 1 to disable, 0 to enable...bitwise filter

class GuideDogCfgPPO( LeggedRobotCfgPPO ):
    class algorithm( LeggedRobotCfgPPO.algorithm ):
        entropy_coef = 0.01
    class runner( LeggedRobotCfgPPO.runner ):

        force_estimation_timesteps = 25
        num_steps_per_env = 48 #24 # per iteration

        run_name = ''
        experiment_name = 'guide_dog'
        load_run = "estimator1" # -1 = last run
        checkpoint = 1500 # -1 = last saved model