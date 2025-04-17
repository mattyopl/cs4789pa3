import torch
from torch import nn
from torch.distributions.categorical import Categorical
from torch.distributions.normal import Normal
import numpy as np


def layer_init(layer, std=0.5, bias_const=0.0):
    if hasattr(layer, "weight"):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    
    def __init__(self, state_dim, action_dim):
        super().__init__()
        # You may want to use layer_init to initialize the weights
        layers = []
        hiddens = [256, 128]
        layers.append(nn.Linear(state_dim, hiddens[0]))

        for i in range(1, len(hiddens)):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hiddens[i - 1], hiddens[i]))

        layers.append(nn.ReLU())
        layers.append(nn.Linear(hiddens[-1], action_dim))
        for layer in layers:
            layer_init(layer)

        policy_network = nn.Sequential(*layers)

        value_layers = []
        value_layers.append(nn.Linear(state_dim, hiddens_value[0]))
        hiddens_value = [512, 128, 32]
        for i in range(1, len(hiddens_value)):
            value_layers.append(nn.ReLU())
            value_layers.append(nn.Linear(hiddens_value[i - 1], hiddens_value[i]))
        for layer in value_layers:
            layer_init(layer)

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.policy_network = policy_network
        self.value_network = nn.Sequential(*value_layers)

    def action_value(self, state, action=None):
        """
        Returns actions, log probability of the actions, the entropy of the distribution and the value at the states

        :param state: The state
        :param action: If action is None then the action is randomly sampled from the policy distribution. 
                       Otherwise, the log probs are computed from the given action. 
        """ 
        # Hint: Use the Categorical distribution
        states = np.zeros(self.state_dim)
        states[state] = 1
        if action == None:
            action_prob_dist = self.network(states)
            dist = Categorical(action_prob_dist)
            dist.sample()

        return action, dist.log_prob(states), dist.entropy(), self.value(state)

        
    
    @torch.no_grad
    def value(self, state):
        """
        Returns the value of the state

        :param state: 
        """
        return self.value_network(state)

class ContinuousActorCritic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Hint: Use the Normal distribution, and have 
        # a single logstd paramater for each action dim irrespective of state
        self.log_std = nn.Parameter(torch.full((action_dim,), -0.5))
        layers = []
        hiddens = [512, 128, 32]
        layers.append(nn.Linear(state_dim, hiddens[0]))

        for i in range(1, len(hiddens)):
            layers.append(nn.ReLU())
            layers.append(nn.Linear(hiddens[i - 1], hiddens[i]))

        layers.append(nn.ReLU())
        layers.append(nn.Linear(hiddens[-1], 1))
        for layer in layers:
            layer_init(layer)

        self.mean_network = nn.Sequential(*layers)

        value_layers = []
        value_layers.append(nn.Linear(state_dim, hiddens_value[0]))
        hiddens_value = [512, 128, 32]
        for i in range(1, len(hiddens_value)):
            value_layers.append(nn.ReLU())
            value_layers.append(nn.Linear(hiddens_value[i - 1], hiddens_value[i]))
        for layer in value_layers:
            layer_init(layer)
        
        self.value_network = nn.Sequential(*value_layers)

    def action_value(self, state, action=None):
        """
        Returns actions, log probability of the actions, the entropy of the distribution and the value at the states

        :param state: The state
        :param action: If action is None then the action is randomly sampled from the policy distribution. 
                       Otherwise, the log probs are computed from the given action. 
        """ 
        mean = self.mean_network(state)
        std = torch.exp(self.log_std)
        distribution = Normal(mean, std)
        return distribution.sample(), distribution.log_prob(state), distribution.entropy(), self.value(state)

    
    @torch.no_grad
    def value(self, state):
        return self.value_network(state)
    