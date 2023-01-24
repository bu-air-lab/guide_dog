import numpy as np

import torch
import torch.nn as nn

class ForceEstimator(nn.Module):
    def __init__(self,  num_obs,
                        force_estimator_hidden_dims=[128,128],
                        activation='elu',
                        init_noise_std=1.0):
        
        super(ForceEstimator, self).__init__()

        activation = nn.ELU()

        #Given obs has zeros in place of all dimensions of estimated state
        mlp_input_dim_se = num_obs - 6
        num_estimated_force_dimensions = 3

        force_estimator_layers = []
        force_estimator_layers.append(nn.Linear(mlp_input_dim_se, force_estimator_hidden_dims[0]))
        force_estimator_layers.append(activation)
        for l in range(len(force_estimator_hidden_dims)):
            if l == len(force_estimator_hidden_dims) - 1:
                force_estimator_layers.append(nn.Linear(force_estimator_hidden_dims[l], num_estimated_force_dimensions))
            else:
                force_estimator_layers.append(nn.Linear(force_estimator_hidden_dims[l], force_estimator_hidden_dims[l + 1]))
                force_estimator_layers.append(activation)
        self.force_estimator = nn.Sequential(*force_estimator_layers)

        print(f"Force Estimator MLP: {self.force_estimator}")


    def forward(self, obs):
        
        #Force estimator doesn't take trailing zeros
        original_obs = obs[:, :-6]
        #original_obs = obs[:, :-3]

        #Query force estimator
        estimated_force = self.force_estimator(original_obs)


        return estimated_force