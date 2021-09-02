import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical
import torch.autograd as autograd
import numpy as np

class Normal_NN(nn.Module):
    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Normal_NN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class Dueling_NN(nn.Module):
    def __init__(self, state_size, action_size, seed, units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Dueling_NN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.feauture_layer = nn.Sequential(
            nn.Linear(state_size, units),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(units, units),
            nn.ReLU(),
            nn.Linear(units, action_size)
        )

    def forward(self, state):
        """Build a network that maps state -> action values."""
        features = self.feauture_layer(state)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())
        return qvals


class Conv_NN(nn.Module):
    def __init__(self, state_dim, action_dim, seed):
        super(Conv_NN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_dim = state_dim
        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim[-1], 32, kernel_size=8, stride=4, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
            nn.ReLU()
        )

        self.fc1 = nn.Linear(self.feature_size(), 512)
        self.fc2 = nn.Linear(512, action_dim)

    def forward(self, x):
        x = x / 255.
        x = self.conv(x)
        x = F.relu(self.fc1(x.reshape(x.size(0), -1)))
        return self.fc2(x)

    @staticmethod
    def init_weights(module):
        if isinstance(module, nn.Linear):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
            module.bias.data.fill_(0.0)
        elif isinstance(module, nn.Conv2d):
            torch.nn.init.kaiming_normal_(module.weight, nonlinearity="relu")

    def feature_size(self):
        return self.conv(torch.zeros(1, *self.input_dim).permute(0,3,1,2)).reshape(1, -1).size(1)


class ConvDueling_NN(nn.Module):

    def __init__(self, state_dim, action_dim, seed):
        super(ConvDueling_NN, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.input_dim = state_dim

        self.conv = nn.Sequential(
            nn.Conv2d(self.input_dim[-1], 32, kernel_size=8, stride=4, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(self.feature_size(), 512),
            nn.ReLU(),
            nn.Linear(512, action_dim)
        )

    def forward(self, state):
        features = self.conv(state)
        features = features.reshape(features.size(0), -1)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        qvals = values + (advantages - advantages.mean())

        return qvals

    def feature_size(self):
        return self.conv(torch.zeros(1, *self.input_dim).permute(0,3,1,2)).reshape(1, -1).size(1)


class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.critic = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

        self.actor = nn.Sequential(
            nn.Linear(num_inputs, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, num_outputs),
            nn.Softmax(dim=1),
        )

    def forward(self, x):
        value = self.critic(x)
        probs = self.actor(x)
        dist  = Categorical(probs)
        return dist, value