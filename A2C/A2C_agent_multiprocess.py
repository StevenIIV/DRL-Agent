import multiprocessing
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from utils.parallel_environments import ParallelEnvironments_info
from NN import ActorCritic_NN
import torch.nn.functional as F
from torch.distributions import Categorical
from tensorboardX import SummaryWriter
import gym
from datetime import datetime
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Storage:
    def __init__(self, steps_per_update, num_of_processes):
        self.steps_per_update = steps_per_update
        self.num_of_processes = num_of_processes
        self.reset_storage()

    def reset_storage(self):
        self.values = torch.zeros(self.steps_per_update,self.num_of_processes,1)
        self.rewards = torch.zeros(self.steps_per_update,self.num_of_processes,1)
        self.action_log_probs = torch.zeros(self.steps_per_update,self.num_of_processes,1)
        self.entropies = torch.zeros(self.steps_per_update,self.num_of_processes)
        self.dones = torch.zeros(self.steps_per_update,self.num_of_processes,1)

    def add(self, step, values, rewards, action_log_probs, entropies, dones):
        self.values[step] = values
        self.rewards[step] = rewards
        self.action_log_probs[step] = action_log_probs
        self.entropies[step] = entropies
        self.dones[step] = dones

    def compute_expected_rewards(self, last_values, discount_factor):
        expected_rewards = torch.zeros(self.steps_per_update + 1,self.num_of_processes,1)
        expected_rewards[-1] = last_values
        for step in reversed(range(self.rewards.size(0))):
            expected_rewards[step] = self.rewards[step] + \
                                     expected_rewards[step + 1] * discount_factor * (1.0 - self.dones[step])
        return expected_rewards[:-1]


class A2CAgent:
    def __init__(self, env_name, n_episode=125000, gamma=0.99, lr=1e-4, update_fre=5):
        self.lr = lr
        self.discount_factor = gamma
        self.value_loss_coef = 0.5
        self.entropy_coef = 0.1
        self.max_norm = 0.5
        self.num_of_steps = n_episode
        self.steps_per_update = update_fre
        self.num_of_processes = 3# multiprocessing.cpu_count()
        self.env_name = env_name
        self.parallel_environments = ParallelEnvironments_info(env_name, number_of_processes=self.num_of_processes)
        self.input_dim, self.act_dim = self.parallel_environments.get_state_and_act()
        self.actor_critic = ActorCritic_NN(self.input_dim, self.act_dim).to(device)
        self.optimizer = Adam(self.actor_critic.parameters(), lr=self.lr)
        self.storage = Storage(self.steps_per_update, self.num_of_processes)
        self.writer = SummaryWriter('../A2C/logs/A2C_multiprocess_Agent_'+env_name+'_'+datetime.now().strftime("%Y%m%d%H%M%S"))
        self.writer.add_text("config", 'game={}, alg={}, n_episode={}, gamma={}, lr={}, update_fre={}'
                             .format(env_name, 'A2C', n_episode, gamma, lr, update_fre), 0)
        self.model_path = '../A2C/saves/A2C_multiprocess_Agent_'+env_name+'.pt'
        self.current_observations = None

    def run(self):
        # num of updates per environment
        num_of_updates = self.num_of_steps / self.steps_per_update
        self.current_observations = self.parallel_environments.reset()

        for update in range(int(num_of_updates)):
            self.storage.reset_storage()
            for step in range(self.steps_per_update):
                probs, log_probs, value = self.actor_critic(self.current_observations.to(device))
                actions = self.get_action(probs)
                action_log_probs, entropies = self.compute_action_logs_and_entropies(probs, log_probs)

                states, rewards, dones = self.parallel_environments.step(actions)
                rewards = rewards.view(-1, 1)
                dones = dones.view(-1, 1)
                self.current_observations = states
                self.storage.add(step, value, rewards, action_log_probs, entropies, dones)

            _, _, last_values = self.actor_critic(self.current_observations)
            expected_rewards = self.storage.compute_expected_rewards(last_values, self.discount_factor)
            advantages = torch.tensor(expected_rewards) - self.storage.values
            value_loss = advantages.pow(2).mean()
            policy_loss = -(advantages * self.storage.action_log_probs).mean()

            self.writer.add_scalar('loss/value loss', value_loss.item(), update)
            self.writer.add_scalar('loss/action loss', policy_loss.item(), update)

            self.optimizer.zero_grad()
            loss = policy_loss - self.entropy_coef * self.storage.entropies.mean() + self.value_loss_coef * value_loss
            self.writer.add_scalar('loss/total loss', loss.item(), update)
            loss.backward(retain_graph=True)
            nn.utils.clip_grad_norm(self.actor_critic.parameters(), self.max_norm)
            self.optimizer.step()

            if update % 300 == 0:
                torch.save(self.actor_critic.state_dict(), self.model_path)

            if update % 100 == 0:
                print('Update: {}. Loss: {}'.format(update, loss))
        self.writer.close()

    def compute_action_logs_and_entropies(self, probs, log_probs):
        values, indices = probs.max(1)
        indices = indices.view(-1, 1)
        action_log_probs = log_probs.gather(1, indices)

        entropies = -(log_probs * probs).sum(-1)

        return action_log_probs, entropies

    def get_action(self, probs):
        actions = []
        for i in range(probs.size(0)):
            dist = Categorical(probs[i])
            action = dist.sample().cpu().detach().item()
            actions.append(action)
        return actions

    def save(self):
        torch.save(self.actor_critic.state_dict(), self.model_path)

    def load(self):
        self.actor_critic.load_state_dict(torch.load(self.model_path))

    def test(self):
        self.load()
        env = gym.make(self.env_name)
        state = env.reset()
        for j in range(1000):
            probs, _, _ = self.actor_critic(state)
            action = self.get_action(probs)
            print('state :{} action :{}'. format(state, action))
            env.render()
            next_state, reward, done, _ = env.step(action)
            state = next_state
            print('next_state={}, reward={}, done={}'.format(next_state, reward, done))
            if done:
                break


if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    agent = A2CAgent('CartPole-v0')
    agent.run()