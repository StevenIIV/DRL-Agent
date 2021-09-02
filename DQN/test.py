import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import torch.optim as optim
from utils.Replayer import ReplayBuffer
from DQN.NN import *
import random
import numpy as np
from datetime import datetime
from tensorboardX import SummaryWriter
from utils.atari_wrappers import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class DoubleDQNAgent():
    """与环境相互作用，从环境中学习。"""

    def __init__(self, state_size, action_size, seed, obs_mode, gamma=0.99, lr=5e-4, update_fre=4):
        """初始化智能体对象。

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.obs_mode = obs_mode
        self.BUFFER_SIZE = int(1e4)  # 经验回放的缓冲区的大小
        self.BATCH_SIZE = 64  # 最小训练批数量
        self.GAMMA = gamma  # 折扣率
        self.TAU = 1e-3  # 用于目标函数的柔性策略更新
        self.LR = lr  # 学习率
        self.UPDATE_EVERY = update_fre  # 更新网络的频率
        print('Program running in {}'.format(device))

        # Q-Network
        if obs_mode == "info":
            self.qnetwork_local = Normal_NN(state_size, action_size, seed).to(device)
            self.qnetwork_target = Normal_NN(state_size, action_size, seed).to(device)
        elif obs_mode == 'img':
            self.qnetwork_local_dqn = Conv_NN(state_size, action_size, seed).to(device)
            self.qnetwork_target_dqn = Conv_NN(state_size, action_size, seed).to(device)
            self.qnetwork_local_ddqn = Conv_NN(state_size, action_size, seed).to(device)
            self.qnetwork_target_ddqn = Conv_NN(state_size, action_size, seed).to(device)
        self.dqn_optimizer = optim.Adam(self.qnetwork_local_dqn.parameters(), lr=self.LR)  # 自适应梯度算法
        self.ddqn_optimizer = optim.Adam(self.qnetwork_local_ddqn.parameters(), lr=self.LR)  # 自适应梯度算法
        # print('Q-Network_local:{}\nQ-Network_target:{}'.format(self.qnetwork_local, self.qnetwork_target))

        # 经验回放
        self.memory = ReplayBuffer(action_size, self.BUFFER_SIZE, self.BATCH_SIZE, seed, device)

        # 初始化时间步 (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # 在经验回放中保存经验
        self.memory.add(state, action, reward, next_state, done)

        # 在每个时间步UPDATE_EVERY中学习
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        loss = 0.0
        loss_comp=0.0
        if self.t_step == 0:
            # 如果内存中有足够的样本，取随机子集进行学习
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                loss = self.ddqn_learn(experiences, self.GAMMA)
                loss_comp = self.dqn_learn(experiences, self.GAMMA)
        return loss,loss_comp

    def act(self, state, eps=0.):
        """根据当前策略返回给定状态的操作.

        Params
        ======
            state (array_like): 当前的状态
            eps (float): epsilon, 用于 epsilon-greedy action selection
        """
        # Epsilon-greedy action selection
        if random.random() > eps:
            if self.obs_mode == 'info':
                state = torch.from_numpy(state).float().unsqueeze(0)
            # 将qn更改成评估形式
            self.qnetwork_local_ddqn.eval()
            # 禁用梯度
            with torch.no_grad():
                # 获得动作价值
                action_values = self.qnetwork_local_ddqn(state.to(device))
            # 将qn更改成训练模式
            self.qnetwork_local_ddqn.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def ddqn_learn(self, experiences, gamma):
        """
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get the expected Q value from the evaluation network
        Q_expected = self.qnetwork_local_ddqn(states).gather(1, actions)

        Q_expected_t = self.qnetwork_local_ddqn(states).to(device)
        # Get the optimal action
        optim_acts = torch.from_numpy(np.vstack(np.argmax(Q_expected_t.cpu().data.numpy(), axis=1))).to(device)
        # Get the maximum predicted Q value (next state) from the target network by optimal action
        Q_targets_next = self.qnetwork_target_ddqn(next_states).gather(1, optim_acts)
        # Calculates the Q target in the current state
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.ddqn_optimizer.zero_grad()
        loss.backward()
        # run optimizer
        self.ddqn_optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local_ddqn, self.qnetwork_target_ddqn, self.TAU)
        return float(loss)

    def dqn_learn(self, experiences, gamma):
        """
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get the expected Q value from the evaluation network
        Q_expected = self.qnetwork_local_dqn(states).gather(1, actions)
        # compute and minimize the loss
        # Get the maximum predicted Q value (next state) from the target network
        Q_targets_next = self.qnetwork_target_dqn(next_states).detach().max(1)[0].unsqueeze(1)
        # Calculates the Q target in the current state
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))


        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.dqn_optimizer.zero_grad()
        loss.backward()
        # run optimizer
        self.dqn_optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local_dqn, self.qnetwork_target_dqn, self.TAU)
        return float(loss)

    def soft_update(self, local_model, target_model, tau):
        """:柔性更新模型参数。
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): 插值参数
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            # 柔性更新, 将src中数据复制到self中
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)


timefig = datetime.now().strftime("%Y%m%d%H%M%S")

model_path = '../DQN/saves/breakout-comp.pth'
log_path1 = '../DQN/logs/breakout-dqn'
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GAMMA = 0.99  # 折扣率
LR = 5e-4  # 学习率
UPDATE_EVERY = 4  # 更新网络的频率

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


def dqn_train(n_episode=10000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """
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
    step = 0
    write_dqn = SummaryWriter('../DQN/logs/breakout-dqn')
    write_ddqn = SummaryWriter('../DQN/logs/breakout-ddqn')
    for i_episode in range(1, n_episode+1):
        # init state
        state = get_state(env.reset())
        score = 0
        for t in range(max_t):
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step(action)
            if done:
                break
            else:
                next_state = get_state(next_state)
            step += 1
            loss,loss_comp = agent.step(state, action, reward, next_state, done)

            write_ddqn.add_scalar('loss', loss, step)
            write_dqn.add_scalar('loss', loss_comp, step)

            state = next_state
            score += reward
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        print('\rEpisode {}\t Average Score:{:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\rAverage Score :{:.2f}'.format(i_episode, np.mean(scores_window)))

    writer.close()
    return scores

import os
if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    env = create_atari_env('Breakout-v0', episode_life=False, frame_stack=4, scale=True, clip_rewards=False)
    state_size = env.observation_space.shape
    action_size = env.action_space.n

    print('game name: ', 'Breakout-v0')
    print('State shape: ', state_size)
    print('Number of actions: ', action_size)

    agent = DoubleDQNAgent(state_size=state_size, action_size=action_size, seed=1, obs_mode='img', gamma=GAMMA, lr=LR, update_fre=UPDATE_EVERY)


    # 训练模式
    env.seed(0)
    scores = dqn_train(n_episode=5000, eps_end=0.01)
    # plot the scores
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # plt.plot(np.arange(len(scores)), scores)
    # plt.ylabel('Score')
    # plt.xlabel('Episode #')
    # plt.show()