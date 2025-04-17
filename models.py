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
        pass

    def action_value(self, state, action=None):
        """
        Returns actions, log probability of the actions, the entropy of the distribution and the value at the states

        :param state: The state
        :param action: If action is None then the action is randomly sampled from the policy distribution. 
                       Otherwise, the log probs are computed from the given action. 
        """ 
        # Hint: Use the Categorical distribution
        raise NotImplementedError()
    
    @torch.no_grad
    def value(self, state):
        """
        Returns the value of the state

        :param state: 
        """
        raise NotImplementedError()

class ContinuousActorCritic(nn.Module):

    def __init__(self, state_dim, action_dim):
        super().__init__()
        # Hint: Use the Normal distribution, and have 
        # a single logstd paramater for each action dim irrespective of state
        raise NotImplementedError()

    def action_value(self, state, action=None):
        """
        Returns actions, log probability of the actions, the entropy of the distribution and the value at the states

        :param state: The state
        :param action: If action is None then the action is randomly sampled from the policy distribution. 
                       Otherwise, the log probs are computed from the given action. 
        """ 
        raise NotImplementedError()
    
    @torch.no_grad
    def value(self, state):
        raise NotImplementedError()
    