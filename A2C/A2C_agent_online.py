import numpy as np
import torch
import gym
from torch import nn
import matplotlib.pyplot as plt
from NN import CriticNetwork, ActorNetwork
import os
from tensorboardX import SummaryWriter
from datetime import datetime
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def t(x): return torch.from_numpy(x).float()


class A2CAgent():
    def __init__(self, env_name, n_episode=5000, gamma=0.99, lr=1e-3, update_fre=1):
        self.env = gym.make(env_name)
        state_dim = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.n
        self.gamma = gamma
        self.lr = lr
        self.entropy_coef = 0.1
        self.value_loss_coef = 0.5
        self.n_episode = n_episode
        self.actor = ActorNetwork(state_dim, n_actions).to(device)
        self.critic = CriticNetwork(state_dim).to(device)
        self.adam_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.adam_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr)
        self.model_path = '../A2C/saves/Online_Agent_'+env_name+'_'+datetime.now().strftime("%Y%m%d%H%M%S")
        self.writer = SummaryWriter('../A2C/logs/Online_Agent_'+env_name)
        self.writer.add_text("config", 'game={}, alg={}, n_episode={}, gamma={}, lr={}, update_fre={}'
                             .format(env_name, 'A2C', n_episode, gamma, lr, update_fre), 0)
        self.steps = 0

    def run(self):
        episode_rewards = []
        for epoch in range(self.n_episode):
            done = False
            total_reward = 0
            state = self.env.reset()
            loss_list = []
            while not done:
                self.steps += 1
                probs, log_probs = self.actor(t(state).to(device))
                entropies = -(log_probs * probs).sum(-1)
                dist = torch.distributions.Categorical(probs=probs)
                action = dist.sample()

                next_state, reward, done, info = self.env.step(action.detach().data.numpy())
                advantage = reward + (1-done)*self.gamma*self.critic(t(next_state)) - self.critic(t(state))

                total_reward += reward
                state = next_state

                critic_loss = advantage.pow(2).mean()
                self.adam_critic.zero_grad()
                critic_loss.backward()
                self.adam_critic.step()
                self.writer.add_scalar('loss/value loss', critic_loss.item(), self.steps)

                actor_loss = -dist.log_prob(action)*advantage.detach()
                self.adam_actor.zero_grad()
                actor_loss.backward()
                self.adam_actor.step()
                self.writer.add_scalar('loss/action loss', actor_loss.item(), self.steps)
                total_loss = actor_loss - self.entropy_coef * entropies.mean() + self.value_loss_coef * critic_loss
                self.writer.add_scalar('loss/total loss', total_loss.item(), self.steps)
                loss_list.append(total_loss.item())

            episode_rewards.append(total_reward)
            self.writer.add_scalar('score', total_reward, epoch)
            self.writer.add_scalar('loss', np.mean(loss_list), epoch)
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
            probs = self.actor(t(state))
            dist = torch.distributions.Categorical(probs=probs)
            action = dist.sample()
            print('state :{} action :{}'. format(state, action))
            self.env.render()
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            print('next_state={}, reward={}, done={}'.format(next_state, reward, done))
            if done:
                break