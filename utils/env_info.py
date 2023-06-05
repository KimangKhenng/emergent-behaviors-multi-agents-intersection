import random
import matplotlib.pyplot as plt

from envs.multi_agents import MultiAgentsInterEnv

import cv2


def get_env_info_single(env):
    # obs = env.reset()

    for step in range(100):
        action = env.action_space.sample()
        o, r, d, i = env.step(action)
    print("State Length: ", o['state'].shape)
    print("Image Shape: ", o['image'].shape)
    print("Env Info: ", i)
    # Plot the image
    plt.imshow(o['image'][:, :, :, -1])

    # Show the plot
    plt.show()


def get_env_info_marl(env):
    obs = env.reset()

    action = env.action_space.sample()
    for a in action.values():
        a[-1] = 1.0
    obs, r, d, i = env.step(action)

    print("State Length: ", obs['agent0']['state'].shape)
    print("Image Shape: ", obs['agent1']['image'].shape)
    print("Env Info: ", i['agent0'])
    cv2.imshow('img', obs['agent3']['image'][..., -1])

    # Show the plot
    cv2.waitKey(0)


if __name__ == '__main__':
    env = MultiAgentsInterEnv(num_agents=8)
    # env = SingleAgentInterEnv()
    get_env_info_marl(env)
