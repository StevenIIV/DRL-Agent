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
parser.add_argument('-gamma', '--gamma', type=float, default=0.99, help='gamma')
parser.add_argument('-eps', '--epsilon', type=float, default=0.01, help='epsilon')

args = parser.parse_args()

timefig = datetime.now().strftime("%Y%m%d%H%M%S")

env_lists = ['CartPole-v0',  'LunarLander-v2', 'Breakout-v0', 'CarRacing-v0']
dqn_lists = ['dqn', 'double_dqn', 'dueling_dqn']

model_path = '../DQN/saves/'+env_lists[args.game_idx]+'-'+dqn_lists[args.dqn_type]+'.pth'
log_path = '../DQN/logs/'+env_lists[args.game_idx]+'_'+dqn_lists[args.dqn_type]+'_'+timefig
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
GAMMA = args.gamma  # 折扣率
#LR = 5e-4  # 学习率
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
    writer = SummaryWriter(log_path)
    writer.add_text("config", 'game={}, alg={}, n_episode={}, gamma={}, lr={}, update_fre={}'
                    .format(env_lists[env_idx], agent.__class__.__name__, n_episode, GAMMA, args.lr, UPDATE_EVERY), 0)
    step = 0
    if obs_mode == 'info':
        writer.add_graph(agent.qnetwork_local, torch.from_numpy(env.reset()).float().unsqueeze(0).to(device))
        for i_episode in range(1, n_episode+1):
            # init state
            state = env.reset()
            score = 0
            for t in range(max_t):
                step += 1
                action = agent.act(state, eps)
                next_state, reward, done, _ = env.step(action)
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

    elif obs_mode == 'img':
        writer.add_graph(agent.qnetwork_local, get_state(env.reset()).to(device))
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
                torch.save(agent.qnetwork_local.state_dict(), model_path)

    writer.close()
    return scores


if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    env_idx = args.game_idx
    obs_mode, _, _, _ = get_game_info(env_lists[env_idx])

    if obs_mode == 'info':
        env = gym.make(env_lists[env_idx])
        state_size = env.observation_space.shape[0]
        action_size = env.action_space.n
    else:
        env = create_atari_env(env_lists[env_idx], episode_life=False, frame_stack=4, scale=True, clip_rewards=False)
        state_size = env.observation_space.shape
        action_size = env.action_space.n

    print('game name: ', env_lists[env_idx])
    print('dqn type:', dqn_lists[args.dqn_type])
    print('State shape: ', state_size)
    print('Number of actions: ', action_size)

    if args.dqn_type == 0:
        agent = DQNAgent(state_size=state_size, action_size=action_size, seed=1, obs_mode=obs_mode, gamma=GAMMA, lr=args.lr, update_fre=UPDATE_EVERY)
    elif args.dqn_type == 1:
        agent = DoubleDQNAgent(state_size=state_size, action_size=action_size, seed=1, obs_mode=obs_mode, gamma=GAMMA, lr=args.lr, update_fre=UPDATE_EVERY)
    else:
        agent = DuelingDQNAgent(state_size=state_size, action_size=action_size, seed=1, obs_mode=obs_mode, gamma=GAMMA, lr=args.lr, update_fre=UPDATE_EVERY)

    if args.mode == 'run':
        tmp = '../saves/'+env_lists[args.game_idx]+'-'+dqn_lists[args.dqn_type]+'.pth'
        agent.qnetwork_local.load_state_dict(torch.load(tmp, map_location=torch.device(device)))
        tmp_writer = SummaryWriter('../test/LunarLander_agent_run')
        if obs_mode == 'info':
            for i in range(1,42):
                state = env.reset()
                score = 0
                for j in range(200):
                    action = agent.act(state)
                    #print('state :{} action :{}'. format(state, action))
                    env.render()
                    next_state, reward, done, _ = env.step(action)
                    score += reward
                    #print('step={}, next_state={}, reward={}, done={}'.format(j, next_state, reward, done))
                    state = next_state
                    #agent.step(state, action, reward, next_state, done)
                    if done:
                        break
                tmp_writer.add_scalar('score', score, global_step=i)
                print('score :{}'. format(score))
        elif obs_mode == 'img':
            for i in range(10):
                state = get_state(env.reset())
                score = 0
                for j in range(200):
                    action = agent.act(state)
                    #print('state :{} action :{}'. format(state, action))
                    env.render()
                    next_state, reward, done, _ = env.step(action)
                    score += reward
                    #print('step={}, next_state={}, reward={}, done={}'.format(j, next_state, reward, done))
                    state = get_state(next_state)
                    #agent.step(state, action, reward, next_state, done)
                    if done:
                        break
                print('score :{}'. format(score))

        env.close()
        tmp_writer.close()

    elif args.mode == 'train':
        # 训练模式
        env.seed(0)
        scores = dqn_train(n_episode=args.episode, eps_end=args.epsilon)
        # plot the scores
        # fig = plt.figure()
        # ax = fig.add_subplot(111)
        # plt.plot(np.arange(len(scores)), scores)
        # plt.ylabel('Score')
        # plt.xlabel('Episode #')
        # plt.show()