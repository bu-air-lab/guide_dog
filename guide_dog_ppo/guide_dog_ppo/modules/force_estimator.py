import numpy as np

import torch
import torch.nn as nn

class ForceEstimator(nn.Module):

    #num_obs is dimension of state
    #num_timesteps is number of total timesteps of states used to predict force
    def __init__(self, num_obs, num_timesteps, hidden_dim=32, train_vel_only=False):

        super(ForceEstimator, self).__init__()

        self.num_obs = num_obs
        self.num_timesteps = num_timesteps
        self.train_vel_only = train_vel_only
        self.activation = nn.ELU()

        #Reduce dimensions of state from 42 -> hidden_dim
        self.mlp_in = nn.Linear(num_obs-6, hidden_dim)
        if(self.train_vel_only):
            self.mlp_in = nn.Linear(6, hidden_dim)

        #Convolve over time dimension. Use parameters from https://arxiv.org/pdf/2107.04034.pdf
        self.conv1 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=7, stride=3)
        self.conv2 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=5, stride=1)
        self.conv3 = nn.Conv1d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, stride=1)

        #Project flattened CNN output to final layer, which outputs force estimate
        self.mlp_out = nn.Linear(hidden_dim, 3)


    def forward(self, obs):
        
        #obs is num environments x num states x num features
        #Force estimator doesn't take trailing zeros
        if(self.train_vel_only):
            original_obs = obs
        else:
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
