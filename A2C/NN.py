import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


class TwoHeadNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(TwoHeadNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim, 64)
        self.linear2 = nn.Linear(64, 64)

        self.actor = nn.Linear(64, output_dim)

        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.relu(x)

        action_probs = F.softmax(self.actor(x))
        state_values = self.critic(x)

        return action_probs, state_values


# Actor module, categorical actions only
class ActorNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, n_units1=64, n_units2=64):
        super(ActorNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, n_units1),
            nn.ReLU(),
            nn.Linear(n_units1, n_units2),
            nn.ReLU(),
            nn.Linear(n_units2, n_actions)
        )

    def forward(self, X):
        X = self.model(X)
        prob = F.softmax(X)
        log_prob = F.log_softmax(X)
        return prob, log_prob


class CriticNetwork(nn.Module):
    def __init__(self, state_dim, n_units1=64, n_units2=64):
        super(CriticNetwork, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, n_units1),
            nn.ReLU(),
            nn.Linear(n_units1, n_units2),
            nn.ReLU(),
            nn.Linear(n_units2, 1)
        )

    def forward(self, X):
        return self.model(X)


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, n_actions, n_units1=64, n_units2=256):
        super(PolicyNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(state_dim, n_units1)
        self.fc2 = nn.Linear(n_units1, n_units2)
        self.fc_mu = nn.Linear(n_units2, n_actions)
        self.fc_std = nn.Linear(n_units2, n_actions)
        self.tanh = nn.Tanh()
        self.softplus = nn.Softplus()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        mu = 2 * self.tanh(self.fc_mu(x))
        std = self.softplus(self.fc_std(x)) + 1e-3
        return mu, std

    def select_action(self, state):
        with torch.no_grad():
            mu, std = self.forward(state)
            n = Normal(mu, std)
            action = n.sample()
        return np.clip(action.item(), -2., 2.)


class ValueNetwork(nn.Module):
    def __init__(self, state_dim, n_units1=64, n_units2=256):
        super(ValueNetwork, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(state_dim, n_units1)
        self.fc2 = nn.Linear(n_units1, n_units2)
        self.fc3 = nn.Linear(n_units2, 1)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ActorCritic_CNN(nn.Module):
    def __init__(self, num_of_inputs, num_of_actions):
        super().__init__()
        self.num_of_inputs = num_of_inputs
        self.conv = nn.Sequential(
            nn.Conv2d(self.num_of_inputs, 32, kernel_size=8, stride=4, bias=False),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, bias=False),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, bias=False),
            nn.ReLU()
        )
        self.linear1 = nn.Linear(self.feature_size(), 512)
        self.policy = nn.Linear(512, num_of_actions)
        self.value = nn.Linear(512, 1)

    def forward(self, x):
        conv_out = self.conv(x)

        flattened = torch.flatten(conv_out, start_dim=1)  # N x 9*9*32
        linear1_out = self.linear1(flattened)

        policy_output = self.policy(linear1_out)
        value_output = self.value(linear1_out)

        probs = F.softmax(policy_output)
        log_probs = F.log_softmax(policy_output)
        return probs, log_probs, value_output

    def feature_size(self):
        return self.conv(torch.zeros(1, *(self.num_of_inputs,84,84))).reshape(1, -1).size(1)


class ActorCritic_NN(nn.Module):
    def __init__(self, num_of_inputs, num_of_actions):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(num_of_inputs, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        self.policy = nn.Linear(64, num_of_actions)
        self.value = nn.Linear(64, 1)

    def forward(self, x):
        linear_out = self.linear(x)

        policy_output = self.policy(linear_out)
        value_output = self.value(linear_out)

        probs = F.softmax(policy_output)
        log_probs = F.log_softmax(policy_output)
        return probs, log_probs, value_output