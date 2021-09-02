import gym
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from DQN.dqn_agent import DQNAgent
from DQN.ddqn_agent import DoubleDQNAgent
from DQN.duelingdqn_agent import DuelingDQNAgent
import os
from tensorboardX import SummaryWriter
import argparse
from utils.atari_wrappers import *
from datetime import datetime

action_space = [
    (-1, 1, 0.2), (0, 1, 0.2), (1, 1, 0.2), #           Action Space Structuress
    (-1, 1,   0), (0, 1,   0), (1, 1,   0), #        (Steering Wheel, Gas, Break)
    (-1, 0, 0.2), (0, 0, 0.2), (1, 0, 0.2), # Range        -1~1       0~1   0~1
    (-1, 0,   0), (0, 0,   0), (1, 0,   0)
]
timefig = datetime.now().strftime("%Y%m%d%H%M%S")
stack_size = 2
mode = 'run'
GAMMA = 0.99  # 折扣率
LR = 5e-4  # 学习率
UPDATE_EVERY = 4  # 更新网络的频率
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


def dqn(n_episode=10000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """
    Deep Q-Learning

    :param n_episode:maximum number of training episodes
    :param max_t:maximum number of timesteps per episode
    :param eps_start:starting value of epsilon, for epsilon-greedy action selection
    :param eps_end:minimum value of epsilon
    :param eps_decay:multiplicative factor (per episode) for decreasing epsilon
    :return: final score
    """
    scores = []
    scores_window = deque(maxlen=100)
    eps = eps_start
    writer = SummaryWriter('../DQN/logs/CarRacingdqn_'+timefig)
    writer.add_text("config", 'game={}, alg={}, n_episode={}, gamma={}, lr={}, update_fre={}, stack_size={}'
                    .format("CarRacing-v0", agent.__class__.__name__, n_episode, GAMMA, LR, UPDATE_EVERY, stack_size), 0)
    step = 0
    for i_episode in range(1, n_episode+1):
        # 初始化状态
        state = get_state(env.reset())
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action_space[action])
            if done:
                break
            else:
                next_state = get_state(next_state)
            step += 1
            loss = agent.step(state, action, reward, next_state, done)
            writer.add_scalar('loss', loss, step)
            state = next_state
            score += reward

        writer.add_scalar('score', score, global_step=i_episode)
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        print('\rEpisode {}\t Average Score:{:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\rAverage Score :{:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), '../DQN/saves/CarRacing'+timefig+'.pth')
        if i_episode == n_episode:
            print('\nEnvironment solved in {:d} episode! \t Average Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), '../DQN/saves/CarRacing'+timefig+'.pth')
            break
    writer.close()
    return scores


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    env = gym.make("CarRacing-v0")
    env = WarpFrame(env, width=96, height=96)
    env = FrameStack(env, stack_size)
    print('game name: ', 'CarRacing')
    print('State shape: ', env.observation_space.shape)
    print('Number of actions: ', action_space)
    state_size = env.observation_space.shape
    action_size = len(action_space) #12


    agent = DoubleDQNAgent(state_size=state_size, action_size=action_size, seed=1, obs_mode='img',
                           gamma=GAMMA, lr=LR, update_fre=UPDATE_EVERY)
    # 训练模式
    if mode == 'train':
        scores = dqn()
        # plot the scores
        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.plot(np.arange(len(scores)), scores)
        plt.ylabel('Score')
        plt.xlabel('Episode #')
        plt.show()

    elif mode == 'run':
        agent.qnetwork_local.load_state_dict(torch.load('../DQN/saves/CarRacing_DoubleDQNAgent.pth', map_location=torch.device(device)))
        state = get_state(env.reset())
        score = 0
        for j in range(1000):
            action = agent.act(state)
            #print('state :{} action :{}'. format(state, action))
            env.render()
            next_state, reward, done, _ = env.step(action_space[action])
            score += reward
            #print('next_state={}, reward={}, done={}'.format(next_state, reward, done))
            state = get_state(next_state)
            #agent.step(state, action, reward, next_state, done)
            if done:
                break
        print('score :{}'. format(score))