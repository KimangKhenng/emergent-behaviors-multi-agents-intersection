import random
import matplotlib.pyplot as plt

from envs.multi_agents import MultiAgentsInterEnv
from envs.multi_agents_lidar import MultiAgentsLidarEnv
from envs.single_agent_intersection_lidar import SingleAgentInterLidarEnv

import cv2


def get_env_info_single(env):
    obs = env.reset()

    # for step in range(1000):
    #     action = env.action_space.sample()
    #     obs, r, d, i = env.step(action)
    print("Obs: ", obs)
    print("State Length: ", obs.shape)
    # print("Env Info: ", i)
    print("Action Space", env.action_space.shape)
    # print("Image Shape: ", o['image'].shape)
    # print("Env Info: ", i)
    # # Plot the image
    # plt.imshow(o['image'][:, :, :, -1])
    #
    # # Show the plot
    # plt.show()


def get_env_info_marl(env):
    obs = env.reset()

    action = env.action_space.sample()
    for a in action.values():
        a[-1] = 1.0
    obs, r, d, i = env.step(action)

    print("State Length: ", obs['agent0'].shape)
    print("Env Info: ", i['agent0'])


if __name__ == '__main__':
    env = MultiAgentsLidarEnv()
    # env = SingleAgentInterLidarEnv()
    # get_env_info_single(env)
    get_env_info_marl(env)
