import random
import torch.optim as optim
from utils.Replayer import ReplayBuffer
from DQN.NN import *
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
class DQNAgent():
    def __init__(self, state_size, action_size, seed, obs_mode, gamma=0.99, lr=5e-4, update_fre=4):
        """
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
        self.BUFFER_SIZE = int(1e4)  # Size of buffer for experience playback
        self.BATCH_SIZE = 64
        self.gamma = gamma # discount factor
        self.TAU = 1e-3  # Flexible policy update for the objective function
        self.LR = lr  # learning rate
        self.UPDATE_EVERY = update_fre  # network update frequency
        print('Program running in {}'.format(device))

        # Q-Network
        if obs_mode == "info":
            self.qnetwork_local = Normal_NN(state_size, action_size, seed).to(device)
            self.qnetwork_target = Normal_NN(state_size, action_size, seed).to(device)
        elif obs_mode == 'img':
            self.qnetwork_local = Conv_NN(state_size, action_size, seed).to(device)
            self.qnetwork_target = Conv_NN(state_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=self.LR)
        # print('Q-Network_local:{}\nQ-Network_target:{}'.format(self.qnetwork_local, self.qnetwork_target))
        # experience replay
        self.memory = ReplayBuffer(action_size, self.BUFFER_SIZE, self.BATCH_SIZE, seed, device)
        # init step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # store into memory
        self.memory.add(state, action, reward, next_state, done)

        # learn in each UPDATE_EVERY
        self.t_step = (self.t_step + 1) % self.UPDATE_EVERY
        loss = 0.0
        if self.t_step == 0:
            if len(self.memory) > self.BATCH_SIZE:
                experiences = self.memory.sample()
                loss = self.learn(experiences, self.gamma)
        return loss

    def act(self, state, eps=0.):
        """
        Params
        ======
            state (array_like): current astate
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # Epsilon-greedy action selection
        if random.random() > eps:
            if self.obs_mode == 'info':
                state = torch.from_numpy(state).float().unsqueeze(0)
            self.qnetwork_local.eval()
            with torch.no_grad():
                # get action value
                action_values = self.qnetwork_local(state.to(device))
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
        # compute and minimize the loss
        # Get the maximum predicted Q value (next state) from the target network
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
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
        """
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): TAU
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)

