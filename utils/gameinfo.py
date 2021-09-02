import numpy as np
import gym
def get_game_info(game_name):
    '''决定state是否为图片，observation&box=image, information&discrete=non-image'''
    env = gym.make(game_name)
    try:
        try:
            shape = env.observation_space.shape
            state_size = (shape[0], shape[1], shape[2])
            state_mode = "img"
        except:
            state_size = env.observation_space.shape[0]
            state_mode = "info"
        state_container = "box"
    except:
        state_size = env.observation_space.n
        state_mode = "info"
        state_container = "discrete"

    '''决定action是否为离散'''
    try:
        action_size = env.action_space.shape[0]
        action_mode = "continuous"
    except:
        action_size = env.action_space.n
        action_mode = "discrete"


    return state_mode, state_size, action_mode, action_size