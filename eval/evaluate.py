import os
import glob
import time
from datetime import datetime

import torch
import numpy as np

from algos.single.ppo_clip_mlp_beta import SPPOClipMLPBeta
# from algos.single.ppo_clip_beta import SinglePPOClipBetaAgent
from algos.single.ppo_clip_mlp_normal import SinglePPOClipMLPNormalAgent
# from envs.single_agent_intersection import SingleAgentInterEnv, STATE_DIM
from envs.single_agent_intersection_lidar import SingleAgentInterLidarEnv
import matplotlib.pyplot as plt


#################################### Testing ###################################
def test():
    print("============================================================================================")

    env_name = "PPO-Clip_Beta-Multi-Intersection-4"
    has_continuous_action_space = True
    max_ep_len = 1000  # max timesteps in one episode
    action_std = 0.1  # set same std for action distribution which was used while saving

    render = True  # render environment on screen
    frame_delay = 0  # if required; add delay b/w frames

    total_test_episodes = 100  # total num of testing episodes

    K_epochs = 5  # update policy for K epochs
    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor
    lr_critic = 0.001  # learning rate for critic

    hidden_size = 256

    #####################################################

    env = SingleAgentInterLidarEnv()

    state_dim = env.observation_space.shape[0]
    # action space dimension
    if has_continuous_action_space:
        action_dim = env.action_space.shape[0]
    else:
        action_dim = env.action_space.n

    # initialize a PPO agent
    # ppo_agent = SinglePPOClipMLPNormalAgent(
    #     state_dim,
    #     action_dim,
    #     lr_actor,
    #     lr_critic,
    #     gamma,
    #     K_epochs,
    #     eps_clip,
    #     has_continuous_action_space,
    #     action_std)

    ppo_agent = SPPOClipMLPBeta(state_dim=state_dim,
                                lr_actor=lr_actor,
                                lr_critic=lr_critic,
                                gamma=gamma,
                                K_epochs=K_epochs,
                                eps_clip=eps_clip,
                                hidden_dim=hidden_size,
                                action_dim=action_dim,
                                )

    # preTrained weights directory

    random_seed = 0  #### set this to load a particular checkpoint trained on random seed
    run_num_pretrained = 0  #### set this to load a particular checkpoint num

    directory = "./PPO_preTrained" + '/' + env_name + '/'
    checkpoint_path = directory + "PPO_{}_{}_{}.pth".format(env_name, random_seed, run_num_pretrained)
    print("loading network from : " + checkpoint_path)

    ppo_agent.load(checkpoint_path)

    print("--------------------------------------------------------------------------------------------")

    test_running_reward = 0

    success_count = 0
    crash_count = 0
    total_episodes = 0
    total_time_to_destination = 0.0
    total_distance_traveled = 0.0
    success_rate = []
    safety_rate = []
    average_velocity_list = []
    time_to_destination_list = []

    for ep in range(1, total_test_episodes + 1):
        ep_reward = 0
        obs = env.reset()

        for t in range(1, max_ep_len + 1):
            action = ppo_agent.select_action(obs)
            obs, r, d, i = env.step(action)
            # print("Action: ", action)
            ep_reward += r
            if i['arrive_dest'] and not i['crash_vehicle'] and not i['out_of_road']:
                success_count += 1

            if i['crash']:
                crash_count += 1

            if render:
                env.render(mode="top_down", film_size=(1000, 1000), track_target_vehicle=True, screen_size=(1000, 1000))
                time.sleep(frame_delay)

            if d:
                total_episodes += 1
                # Calculate success rate and safety rate
                success_rate.append(success_count / total_episodes)
                safety_rate.append((total_episodes - crash_count) / total_episodes)
                break

        # clear buffer
        # ppo_agent.rollout_buffer.clear()

        # Update total episode count

        test_running_reward += ep_reward
        print('Episode: {} \t\t Reward: {}'.format(ep, round(ep_reward, 2)))
        ep_reward = 0

    env.close()

    print("============================================================================================")

    avg_test_reward = test_running_reward / total_test_episodes
    avg_test_reward = round(avg_test_reward, 2)
    print("average test reward : " + str(avg_test_reward))

    print("============================================================================================")

    # Plotting success rate
    plt.plot(success_rate)
    plt.xlabel('Episodes')
    plt.ylabel('Success Rate')
    plt.title('Success Rate over Episodes')
    plt.show()

    # Plotting safety rate
    plt.plot(safety_rate)
    plt.xlabel('Episodes')
    plt.ylabel('Safety Rate')
    plt.title('Safety Rate over Episodes')
    plt.show()


if __name__ == '__main__':
    test()
