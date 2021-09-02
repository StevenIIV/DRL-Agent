import torch.optim as optim
from utils.Replayer import ReplayBuffer
from DQN.NN import *
import random
import numpy as np



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
            self.qnetwork_local = Conv_NN(state_size, action_size, seed).to(device)
            self.qnetwork_target = Conv_NN(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)  # 自适应梯度算法
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
        if self.t_step == 0:
            # 如果内存中有足够的样本，取随机子集进行学习
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                loss = self.learn(experiences, self.GAMMA)
        return loss

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
            self.qnetwork_local.eval()
            # 禁用梯度
            with torch.no_grad():
                # 获得动作价值
                action_values = self.qnetwork_local(state.to(device))
            # 将qn更改成训练模式
            self.qnetwork_local.train()
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        """
        Params
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        # Get the expected Q value from the evaluation network
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        Q_expected_t = self.qnetwork_local(states).to(device)
        # Get the optimal action
        optim_acts = torch.from_numpy(np.vstack(np.argmax(Q_expected_t.cpu().data.numpy(), axis=1))).to(device)
        # Get the maximum predicted Q value (next state) from the target network by optimal action
        Q_targets_next = self.qnetwork_target(next_states).gather(1, optim_acts)
        # Calculates the Q target in the current state
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        # run optimizer
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, self.TAU)
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


