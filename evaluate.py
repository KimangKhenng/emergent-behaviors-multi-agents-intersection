import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

from algos.single.ppo_clip_beta import SinglePPOClipBetaAgent
from envs.single_agent_intersection import SingleAgentInterEnv, STATE_DIM


#################################### Testing ###################################
def test():
    print("============================================================================================")

    ################## hyperparameters ##################

    # env_name = "CartPole-v1"
    # has_continuous_action_space = False
    # max_ep_len = 400
    # action_std = None

    # env_name = "LunarLander-v2"
    # has_continuous_action_space = False
    # max_ep_len = 300
    # action_std = None

    # env_name = "BipedalWalker-v2"
    # has_continuous_action_space = True
    # max_ep_len = 1500           # max timesteps in one episode
    # action_std = 0.1            # set same std for action distribution which was used while saving

    env_name = "PPO-Clip_Beta-Single-Agent-Intersection"
    has_continuous_action_space = True
    max_ep_len = 1000  # max timesteps in one episode
    action_std = 0.1  # set same std for action distribution which was used while saving

    render = True  # render environment on screen
    frame_delay = 0  # if required; add delay b/w frames

    total_test_episodes = 1000  # total num of testing episodes

    K_epochs = 5  # update policy for K epochs
    batch_size = 100  # training batch size
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor
    lr_critic = 0.001  # learning rate for critic

    #####################################################

    env = SingleAgentInterEnv()

    # initialize a PPO agent
    ppo_agent = SinglePPOClipBetaAgent(state_size=STATE_DIM,
                                       batch_size=batch_size,
                                       lr_actor=lr_actor,
                                       lr_critic=lr_critic,
                                       gamma=gamma,
                                       k_epochs=K_epochs,
                                       eps_clip=eps_clip)

    # preTrained weights directory

    random_seed = 0  #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0  #### set this to load a particular checkpoint num

    directory = "PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    for ep in range(1, total_test_episodes + 1):
        ep_reward = 0
        obs = env.reset()

        for t in range(1, max_ep_len + 1):
            action = ppo_agent.select_action(obs)
            obs, r, d, i = env.step(action)
            ep_reward += r

            if render:
                env.render(mode="top_down", film_size=(1000, 1000), track_target_vehicle=True, screen_size=(1000, 1000))
                time.sleep(frame_delay)

            if d:
                break

        # clear buffer
        ppo_agent.rollout_buffer.clear()

        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")


if __name__ == '__main__':
    test()
