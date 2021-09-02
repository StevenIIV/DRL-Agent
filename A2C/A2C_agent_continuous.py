import torch
import torch.nn as nn
from itertools import count
from torch.distributions import Normal
import numpy as np
from collections import deque
import random
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from NN import PolicyNetwork, ValueNetwork
import os
import gym
from datetime import datetime
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
def t(x): return torch.from_numpy(x).float()


# Memory
# Stores results from the networks, instead of calculating the operations again from states, etc.
class Memory():
    def __init__(self, device):
        self.device = device
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

    def add(self, log_prob, value, reward, done):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def get(self):
        return torch.stack(self.log_probs).to(self.device), torch.stack(self.values).to(self.device)

    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()

    def _zip(self):
        return zip(self.log_probs,
                   self.values,
                   self.rewards,
                   self.dones)

    def __iter__(self):
        for data in self._zip():
            return data

    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data

    def __len__(self):
        return len(self.rewards)


# unlike A2CAgent in a2c.py, here I separated value and policy network.
class A2CAgent():
    def __init__(self, env_name, n_episode=1000, gamma=0.99, lr=1e-3, update_fre=64):

        self.env = gym.make(env_name)
        self.state_dim = self.env.observation_space.shape[0]
        self.n_actions = self.env.action_space.shape[0]
        self.gamma = gamma
        self.lr = lr
        self.steps = 0
        self.entropy_coef = 0.1
        self.value_loss_coef = 0.5
        self.update_fre = update_fre
        self.actor = PolicyNetwork(self.state_dim, self.n_actions).to(device)
        self.critic = ValueNetwork(self.state_dim).to(device)
        self.adam_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.adam_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.n_episode = n_episode
        self.writer = SummaryWriter('../A2C/logs/Discrete_Agent_'+env_name+'_'+datetime.now().strftime("%Y%m%d%H%M%S"))
        self.writer.add_text("config", 'game={}, alg={}, n_episode={}, gamma={}, lr={}, update_fre={}'
                             .format(env_name, 'A2C', n_episode, gamma, lr, update_fre), 0)
        self.model_path = '../A2C/saves/A2C_Agent_'+env_name


    def train(self, memory, q_val):
        log_probs, values = memory.get()
        q_vals = np.zeros((len(memory), 1))

        # target values are calculated backward
        # it's super important to handle correctly done states,
        # for those cases we want our to target to be equal to the reward only
        for i, (_, _, reward, done) in enumerate(memory.reversed()):
            q_val = reward + self.gamma * q_val*(1.0-done)
            q_vals[len(memory)-1 - i] = q_val # store values from the end to the beginning

        advantage = torch.Tensor(q_vals).to(device) - values

        critic_loss = advantage.pow(2).mean()
        self.adam_critic.zero_grad()
        critic_loss.backward()
        self.adam_critic.step()
        self.writer.add_scalar('loss/value loss', critic_loss.item(), self.steps)

        actor_loss = (-log_probs * advantage.detach()).mean()
        self.adam_actor.zero_grad()
        actor_loss.backward()
        self.adam_actor.step()
        self.writer.add_scalar('loss/action loss', actor_loss.item(), self.steps)


    def run(self):
        memory = Memory(device)
        episode_rewards = []
        for epoch in range(self.n_episode):
            done = False
            total_reward = 0
            state = self.env.reset()
            while not done:
                self.steps += 1
                mu, std = self.actor(t(state).to(device))
                n = Normal(mu, std)
                action = np.clip(n.sample().item(), -2., 2.)

                next_state, reward, done, info = self.env.step([action])

                total_reward += reward
                self.steps += 1
                memory.add(n.log_prob(torch.FloatTensor([action]).to(device)), self.critic(t(state).to(device)), reward, done)

                state = next_state

                # train if done or num steps > max_steps
                if done or (self.steps % self.update_fre == 0):
                    last_q_val = self.critic(t(next_state).to(device)).cpu().detach().data.numpy()
                    self.train(memory, last_q_val)
                    memory.clear()

            episode_rewards.append(total_reward)
            self.writer.add_scalar('score', total_reward, epoch)
            if epoch % 50 == 0:
                print(epoch, np.mean(episode_rewards[-50:]))

    def save(self):
        torch.save(self.actor.state_dict(), self.model_path+'-actor.pt')
        torch.save(self.critic.state_dict(), self.model_path+'-critic.pt')

    def load(self):
        self.actor.load_state_dict(torch.load(self.model_path+'-actor.pt'))
        self.critic.load_state_dict(torch.load(self.model_path+'-critic.pt'))

    def test(self):
        self.load()
        state = self.env.reset()
        for j in range(1000):
            action = self.actor.select_action(t(state))
            print('state :{} action :{}'. format(state, action))
            self.env.render()
            next_state, reward, done, _ = self.env.step([action])
            state = next_state
            print('next_state={}, reward={}, done={}'.format(next_state, reward, done))
            if done:
                break


if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    agent = A2CAgent('Pendulum-v0')
    agent.run()