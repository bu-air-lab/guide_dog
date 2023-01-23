import numpy as np

import torch
import torch.nn as nn

class BaseVelocityEstimator(nn.Module):
    def __init__(self,  num_obs,
                        base_velocity_estimator_hidden_dims=[128,128],
                        activation='elu',
                        init_noise_std=1.0):
        
        super(BaseVelocityEstimator, self).__init__()

        activation = nn.ELU()

        #Given obs has zeros in place of all dimensions of estimated state
        mlp_input_dim_se = num_obs - 6
        num_estimated_base_velocity_dimensions = 3

        base_velocity_estimator_layers = []
        base_velocity_estimator_layers.append(nn.Linear(mlp_input_dim_se, base_velocity_estimator_hidden_dims[0]))
        base_velocity_estimator_layers.append(activation)
        for l in range(len(base_velocity_estimator_hidden_dims)):
            if l == len(base_velocity_estimator_hidden_dims) - 1:
                base_velocity_estimator_layers.append(nn.Linear(base_velocity_estimator_hidden_dims[l], num_estimated_base_velocity_dimensions))
            else:
                base_velocity_estimator_layers.append(nn.Linear(base_velocity_estimator_hidden_dims[l], base_velocity_estimator_hidden_dims[l + 1]))
                base_velocity_estimator_layers.append(activation)
        self.base_velocity_estimator = nn.Sequential(*base_velocity_estimator_layers)

        print(f"Base Velocity Estimator MLP: {self.base_velocity_estimator}")


    def forward(self, obs):
        
        #Base Velocity estimator doesn't take trailing zeros
        original_obs = obs[:, :-6]

        #Query base velocity estimator
        estimated_base_lin_vel = self.base_velocity_estimator(original_obs)


        return estimated_base_lin_vel