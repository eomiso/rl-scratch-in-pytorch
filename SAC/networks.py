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
    def __init__(
            self,
            beta,  # learning rate
            input_dims,
            n_actions,
            name='critic',
            fc1_dims=256,
            fc2_dims=256,
            chkpt_dir='tmp/sac'):
        super().__init__()
        self.beta = beta
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')
        # we want to incorporate action right from the begining
        # the Critic evaluates state-action pair.
        # in deep deterministics policy gradients, you can start we states only

        self.fc1 = nn.Linear(self.input_dims[0] + n_actions, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)
        self.q = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.cuda.device(
            'cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state, action):
        action_value = self.fc1(T.cat([state, action], dim=1))
        action_value = F.relu(action_value)
        action_value = self.fc2(action_value)
        action_value = F.relu(action_value)

        q = self.q(action_value)

        return q

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


class ValueNetwork(nn.Module):
    def __init__(self,
                 beta,
                 input_dims,
                 fc1_dims=256,
                 fc2_dims=256,
                 name='value',
                 chkpt_dir='tmp/sac'):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.checkpoint_dir = chkpt_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, name + '_sac')

        self.fc1 = nn.Linear(*self.input_dims, fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, fc2_dims)
        self.v = nn.Linear(self.fc2_dims, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.cuda.device(
            'cuda:0' if T.cuda.is_available() else 'cpu')

    def forward(self, state):
        state_val = self.fc1(state)
        state_val = F.relu(state_val)
        state_val = self.fc2(state_val)
        state_val = F.relu(state_val)

        v = self.v(state_val)

        return v

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))


# Handle sampling probability distributions instead of just
# doing a simple feed forward
class ActorNetwork(nn.Module):
    def __init__(self,
                 alpha,
                 input_dims,
                 max_action,
                 fc1_dims=256,
                 fc2_dims=256,
                 n_actions=2,
                 name='actor',
                 checkpoint_dir='tmp/sac'):
        super().__init__()
        self.input_dims = input_dims
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.name = name
        self.max_action = max_action
        self.n_actions = n_actions
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, name + '_sac')

        # need reparametrization
        self.reparam_noise = 1e-6

        self.fc1 = nn.Linear(*self.input_dims, self.fc1_dims)
        self.fc2 = nn.Linear(self.fc1_dims, self.fc2_dims)

        # the mean of our distribution for policy
        self.mu = nn.Linear(self.fc2_dims, self.n_actions)
        # the standard deviation of our distribution for policy
        self.sigma = nn.Linear(self.fc2_dims, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.cuda.device(
            'cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        prob = self.fc1(state)
        prob = F.relu(prob)
        prob = self.fc2(prob)
        prob = F.relu(prob)

        mu = self.mu(prob)
        sigma = self.sigma(prob)

        sigma = T.clamp(sigma, min=self.reparam_noise, max=1)

        # clamp the sigma. you don't want the distributions for your
        # policy to be arbitrarily broad

        return mu, sigma

    # Calculate actual action
    # Policy is a probability distribution given some state.
    # if the action space is discrete you would just simply
    # assign probability to each actions.
    def sample_normal(self, state, reparametrize=True):
        mu, sigma = self.forward(state)
        probabilities = Normal(mu, sigma)

        if reparametrize:
            # adding some noise
            # for more exploration
            actions = probabilities.rsample()
        else:
            actions = probabilities.sample()

        action = T.tanh(actions) * T.tensor(self.max_action).to(self.device)
        # goes into the calculation of the loss function
        log_probs = probabilities.log_prob(actions)
        # log zero is not defined, comes from appendix in the paper
        log_probs -= T.log(1 - action.pow(2) + self.reparam_noise)
        # We need a scalar quantity
        log_probs = log_probs.sum(1, keepdim=True)

        return action, log_probs

    def save_checkpoint(self):
        T.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(T.load(self.checkpoint_file))
