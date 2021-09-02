import torch
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from A2C.NN import ActorCritic_CNN
from utils.atari_wrappers import create_atari_env
from tensorboardX import SummaryWriter
import collections
import random
from datetime import datetime
from collections import deque
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Memory
# Stores results from the networks, instead of calculating the operations again from states, etc.
class Memory():
    def __init__(self, device):
        self.device = device
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.entropies = []

    def add(self, log_prob, value, reward, done, entropy):
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.entropies.append(entropy)

    def get(self):
        return torch.stack(self.log_probs).to(self.device), torch.stack(self.values).to(self.device), torch.stack(self.entropies).to(self.device)

    def clear(self):
        self.log_probs.clear()
        self.values.clear()
        self.rewards.clear()
        self.dones.clear()
        self.entropies.clear()

    def _zip(self):
        return zip(self.log_probs,
                   self.values,
                   self.rewards,
                   self.dones,
                   self.entropies)

    def __iter__(self):
        for data in self._zip():
            return data

    def reversed(self):
        for data in list(self._zip())[::-1]:
            yield data

    def __len__(self):
        return len(self.rewards)

def get_state(obs):
    state = np.array(obs)
    # print(state.shape)  # 84*84*4
    state = state.transpose((2, 0, 1))
    state = state.astype(np.float32)
    state = torch.from_numpy(state)
    # plt.imshow(state[3])
    # plt.show()
    # time.sleep(2)
    return state.unsqueeze(0)

# unlike A2CAgent in a2c.py, here I separated value and policy network.
class A2CAgent():
    def __init__(self, env_name, n_episode=1000, gamma=0.99, lr=1e-3, update_fre=64, stack_size=4):
        self.env = create_atari_env(env_name, episode_life=False, frame_stack=stack_size, scale=True, clip_rewards=False)
        self.obs_dim = self.env.observation_space.shape
        self.action_dim = self.env.action_space.n
        self.gamma = gamma
        self.lr = lr
        self.steps = 0
        self.entropy_coef = 0.1
        self.value_loss_coef = 0.5
        self.update_fre = update_fre
        self.actor_critic = ActorCritic_CNN(stack_size, self.action_dim).to(device)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=self.lr)

        self.n_episode = n_episode
        self.writer = SummaryWriter('../A2C/logs/Discrete_Agent_'+env_name+'_'+datetime.now().strftime("%Y%m%d%H%M%S"))
        self.writer.add_text("config", 'game={}, alg={}, n_episode={}, gamma={}, lr={}, update_fre={}'
                             .format(env_name, 'A2C', n_episode, gamma, lr, update_fre), 0)
        self.model_path = '../A2C/saves/Discrete_Agent_'+env_name


    def train(self, memory, q_val):
        log_probs, values, entropies = memory.get()
        q_vals = np.zeros((len(memory), 1))

        # target values are calculated backward
        # it's super important to handle correctly done states,
        # for those cases we want our to target to be equal to the reward only
        for i, (_, _, reward, done, _) in enumerate(memory.reversed()):
            q_val = reward + self.gamma * q_val*(1.0-done)
            q_vals[len(memory)-1 - i] = q_val # store values from the end to the beginning

        advantage = torch.Tensor(q_vals).to(device) - values

        critic_loss = advantage.pow(2).mean()
        actor_loss = (-log_probs * advantage.detach()).mean()
        self.writer.add_scalar('loss/value loss', critic_loss.item(), self.steps)
        self.writer.add_scalar('loss/action loss', actor_loss.item(), self.steps)

        self.optimizer.zero_grad()
        loss = actor_loss - self.entropy_coef * entropies.mean() + self.value_loss_coef * critic_loss
        self.writer.add_scalar('loss/total loss', loss.item(), self.steps)
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def run(self):
        memory = Memory(device)
        episode_rewards = []
        for epoch in range(self.n_episode):
            done = False
            total_reward = 0
            state = get_state(self.env.reset())
            loss_list = []
            while not done:
                self.steps += 1
                probs, log_probs, value = self.actor_critic(state.to(device))
                entropies = -(log_probs * probs).sum(-1)
                dist = Categorical(probs=probs)
                action = dist.sample()

                next_state, reward, done, info = self.env.step(action.cpu().detach().data.numpy())

                total_reward += reward
                self.steps += 1
                memory.add(dist.log_prob(action), value, reward, done, entropies)
                next_state = get_state(next_state)
                state = next_state

                # train if done or num steps > max_steps
                if done or (self.steps % self.update_fre == 0):
                    _, _, last_q_val = self.actor_critic(next_state.to(device))
                    loss_list.append(self.train(memory, last_q_val.cpu().detach()))
                    memory.clear()

            episode_rewards.append(total_reward)
            self.writer.add_scalar('loss', np.mean(loss_list), epoch)
            self.writer.add_scalar('score', total_reward, epoch)
            if epoch % 50 == 0:
                print(epoch, np.mean(episode_rewards[-50:]))

    def save(self):
        torch.save(self.actor_critic.state_dict(), self.model_path)

    def load(self):
        self.actor_critic.load_state_dict(torch.load(self.model_path))


    def test(self):
        self.load()
        state = self.env.reset()
        for j in range(1000):
            probs, log_probs, _ = self.actor_critic(state.to(device))
            dist = Categorical(probs=probs)
            action = dist.sample()
            print('state :{} action :{}'. format(state, action))
            self.env.render()
            next_state, reward, done, _ = self.env.step(action)
            state = next_state
            print('next_state={}, reward={}, done={}'.format(next_state, reward, done))
            if done:
                break

if __name__ == '__main__':
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    agent = A2CAgent('Breakout-v0')
    agent.run()