# We want to solve the problem of robust and stable 
# learning in continuous action space environments

# We may try TD3 as well
# Hear we are going to add an extra parameter(entropy)
# which will robust to random seeds and episode to episode 
# starting conditions.

# we need 3 networks actor, critic, value

import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, beta, input_dims, n_actions, fc1_dims=256, fc2_dims=256, name, chkpt_dir='tmp/sac'):
        super().__init__()
