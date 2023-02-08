import numpy as np

import torch
import torch.nn as nn

class ForceEstimator(nn.Module):

    #num_obs is dimension of state
    #num_timesteps is number of total timesteps of states used to predict force
    def __init__(self, num_obs, num_timesteps):

        super(ForceEstimator, self).__init__()

        self.num_obs = num_obs
        self.num_timesteps = num_timesteps

        self.activation = nn.ELU()

        #Reduce dimensions of state from 42 -> 32
        self.mlp_in = nn.Linear(num_obs-6, 32)

        #Convolve over time dimension. Use parameters from https://arxiv.org/pdf/2107.04034.pdf
        self.conv1 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=7, stride=3)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=5, stride=1)
        self.conv3 = nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=1)

        #Project flattened CNN output to final layer, which outputs force estimate
        self.mlp_out = nn.Linear(32, 3)


    def forward(self, obs):
        
        #obs is num environments x num states x num features
        #Force estimator doesn't take trailing zeros
        original_obs = obs[:, :, :-6]

        #Reduce dimensionality of each state
        x = self.activation(self.mlp_in(original_obs))

        #Apply convolution to capture temporal relations
        #conv1d expects batch x channel x data.
        #We need to swap num states and num features dimensions, 
        #because we want to capture temporal relation across states.
        x = torch.transpose(x, 1, 2)
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))

        #Flatten CNN output
        x = torch.flatten(x, 1)

        estimated_force = self.mlp_out(x)

        return estimated_force


"""
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
"""