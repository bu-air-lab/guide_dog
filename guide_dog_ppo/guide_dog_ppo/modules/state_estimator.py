import numpy as np

import torch
import torch.nn as nn

class StateEstimator(nn.Module):
    def __init__(self,  num_obs,
                        state_estimator_hidden_dims=[128,128],
                        activation='elu',
                        init_noise_std=1.0):
        
        super(StateEstimator, self).__init__()

        activation = get_activation(activation)

        #Given obs has zeros in place of all dimensions of estimated state
        mlp_input_dim_se = num_obs - 6
        num_estimated_state_dimensions = 6

        # State-Estimator
        state_estimator_layers = []
        state_estimator_layers.append(nn.Linear(mlp_input_dim_se, state_estimator_hidden_dims[0]))
        state_estimator_layers.append(activation)
        for l in range(len(state_estimator_hidden_dims)):
            if l == len(state_estimator_hidden_dims) - 1:
                state_estimator_layers.append(nn.Linear(state_estimator_hidden_dims[l], num_estimated_state_dimensions))
            else:
                state_estimator_layers.append(nn.Linear(state_estimator_hidden_dims[l], state_estimator_hidden_dims[l + 1]))
                state_estimator_layers.append(activation)
        self.state_estimator = nn.Sequential(*state_estimator_layers)

        print(f"State-Estimator MLP: {self.state_estimator}")


    def forward(self, obs):
        
        #State estimator doesn't take trailing zeros
        original_obs = obs[:, :-6]

        #Query state estimator
        estimated_base_lin_vel = self.state_estimator(original_obs)


        return estimated_base_lin_vel
    

def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    else:
        print("invalid activation function!")
        return None
