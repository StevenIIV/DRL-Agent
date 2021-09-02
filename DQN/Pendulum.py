import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import os
import torch
from DQN.ddqn_agent import DoubleDQNAgent
from DQN.dqn_agent import DQNAgent
from DQN.duelingdqn_agent import DuelingDQNAgent
from tensorboardX import SummaryWriter
from utils.gameinfo import *
from utils.atari_wrappers import *
import argparse
from datetime import datetime

'''命令项选项'''
parser = argparse.ArgumentParser()
parser.add_argument('-gidx', '--game_idx', type=int, default=0, help='range in 0 to 6')
parser.add_argument('-mode', '--mode', type=str, default='train', help='train or run')
parser.add_argument('-dqn', '--dqn_type', type=int, default=0, help='0:dqn, 1:double_dqn, 2:dueling_dqn')
parser.add_argument('-step', '--episode', type=int, default=1000, help='episode')
parser.add_argument('-lr', '--lr', type=float, default=1e-3, help='lr')
args = parser.parse_args()

timefig = datetime.now().strftime("%Y%m%d%H%M%S")

dqn_lists = ['dqn', 'double_dqn', 'dueling_dqn']

game_name = "Pendulum-v0"
model_path = '../DQN/saves/'+game_name+'-'+dqn_lists[args.dqn_type]+'.pth'
log_path = '../DQN/logs/'+game_name+'_'+dqn_lists[args.dqn_type]+'_'+timefig
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GAMMA = 0.99  # 折扣率
#LR = 5e-4  # 学习率
UPDATE_EVERY = 4  # 更新网络的频率
actions = [-2.0,-1.75,-1.5,-1.25,-1,-0.75,-0.5,-0.25,0,0.25,0.5,0.75,1,1.25,1.5,1.75,2]


def dqn_train(n_episode=10000, max_t=1000, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
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
    writer = SummaryWriter(log_path)
    writer.add_text("config", 'game={}, alg={}, n_episode={}, gamma={}, lr={}, update_fre={}'
                    .format(game_name, agent.__class__.__name__, n_episode, GAMMA, args.lr, UPDATE_EVERY), 0)
    step = 0
    writer.add_graph(agent.qnetwork_local, torch.from_numpy(env.reset()).float().unsqueeze(0).to(device))
    for i_episode in range(1, n_episode+1):
        # 初始化状态
        state = env.reset()
        score = 0
        for t in range(max_t):
            step += 1
            action = agent.act(state, eps)
            next_state, reward, done, _ = env.step([actions[action]])
            loss = agent.step(state, action, reward, next_state, done)
            writer.add_scalar('loss', loss, step)
            state = next_state
            score += reward
            if done:
                break
        writer.add_scalar('score', score, global_step=i_episode)
        scores_window.append(score)
        scores.append(score)
        eps = max(eps_end, eps_decay*eps)
        print('\rEpisode {}\t Average Score:{:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\rAverage Score :{:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), model_path)
        if i_episode == n_episode:
            print('\nEnvironment solved in {:d} episode! \t Average Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), model_path)
            break
    writer.close()
    return scores


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    env = gym.make(game_name)
    state_size = env.observation_space.shape[0]
    action_size = len(actions)

    print('game name: ', game_name)
    print('dqn type:', dqn_lists[args.dqn_type])
    print('State shape: ', state_size)
    print('Number of actions: ', len(actions))

    if args.dqn_type == 0:
        agent = DQNAgent(state_size=state_size, action_size=action_size, seed=1, obs_mode='info', gamma=GAMMA, lr=args.lr, update_fre=UPDATE_EVERY)
    elif args.dqn_type == 1:
        agent = DoubleDQNAgent(state_size=state_size, action_size=action_size, seed=1, obs_mode='info', gamma=GAMMA, lr=args.lr, update_fre=UPDATE_EVERY)
    else:
        agent = DuelingDQNAgent(state_size=state_size, action_size=action_size, seed=1, obs_mode='info', gamma=GAMMA, lr=args.lr, update_fre=UPDATE_EVERY)

    if args.mode == 'run':
        agent.qnetwork_local.load_state_dict(torch.load(model_path, map_location=torch.device(device)))
        for i in range(10):
            state = env.reset()
            score = 0
            for j in range(1000):
                action = agent.act(state)
                #print('state :{} action :{}'. format(state, action))
                env.render()
                next_state, reward, done, _ = env.step([actions[action]])
                score += reward
                #print('step={}, next_state={}, reward={}, done={}'.format(j, next_state, reward, done))
                state = next_state
                #agent.step(state, action, reward, next_state, done)
                if done:
                    break
            print(score)
        env.close()

    elif args.mode == 'train':
        env.seed(0)
        # 训练模式
        scores = dqn_train(n_episode=args.episode)
        # plot the scores
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # plt.plot(np.arange(len(scores)), scores)
        # plt.ylabel('Score')
        # plt.xlabel('Episode #')
        # plt.show()