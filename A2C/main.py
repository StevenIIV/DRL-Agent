import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import A2C.A2C_agent_decoupled as Discrete_Agent
import A2C.A2C_agent_continuous as Continuous_Agent
import A2C.A2C_agent_online as Online_Agent
import A2C.A2C_agent_cnn as Img_Agent
import A2C.A2C_agent_CarRacing as Car_Agent
from utils.gameinfo import get_game_info
from utils.environment_wrapper import EnvironmentWrapper
import argparse
from datetime import datetime

'''命令项选项'''
parser = argparse.ArgumentParser()
parser.add_argument('-gidx', '--game_idx', type=int, default=2, help='range in 0 to 6')
parser.add_argument('-mode', '--mode', type=str, default='train', help='train or run')
parser.add_argument('-policy', '--policy', type=str, default='off', help='offline and online')
parser.add_argument('-step', '--episode', type=int, default=1000, help='episode')
parser.add_argument('-lr', '--lr', type=float, default=1e-3, help='lr')
parser.add_argument('-update', '--update', type=int, default=64, help='lr')
parser.add_argument('-gamma', '--gamma', type=float, default=0.99, help='gamma')

args = parser.parse_args()


env_lists = ['CartPole-v0', 'LunarLander-v2', 'Pendulum-v0', 'Breakout-v0', 'CarRacing-v0']

timefig = datetime.now().strftime("%Y%m%d%H%M%S")

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
    state_mode, state_size, action_mode, action_size = get_game_info(env_lists[args.game_idx])
    print('game name: ', env_lists[args.game_idx])
    print('alg type: A2C')
    print('State shape: ', state_size)
    print('Number of actions: ', action_size)
    agent = None

    if env_lists[args.game_idx] == 'CarRacing-v0':
        agent = Car_Agent.A2CAgent(env_lists[args.game_idx], n_episode=args.episode, lr=args.lr, gamma=args.gamma)
    else:
        if action_mode == 'continuous':
            agent = Continuous_Agent.A2CAgent(env_lists[args.game_idx], n_episode=args.episode, lr=args.lr, update_fre=args.update, gamma=args.gamma)
        elif action_mode == 'discrete':
            if args.policy == 'off':
                if state_mode == 'img':
                    agent = Img_Agent.A2CAgent(env_lists[args.game_idx], n_episode=args.episode, lr=args.lr, update_fre=args.update, gamma=args.gamma)
                else:
                    agent = Discrete_Agent.A2CAgent(env_lists[args.game_idx], n_episode=args.episode, lr=args.lr, update_fre=args.update, gamma=args.gamma)
            elif args.policy == 'on':
                agent = Online_Agent.A2CAgent(env_lists[args.game_idx], n_episode=args.episode, lr=args.lr)

    if args.mode == 'train':
        agent.run()
        agent.save()
    elif args.mode == 'run':
        agent.test()





