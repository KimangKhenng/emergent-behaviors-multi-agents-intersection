import random

from envs.multi_agents import MultiAgentsEnv
from envs.single_agent_intersection import SingleAgentsEnv
import matplotlib.pyplot as plt
from metadrive import MetaDriveEnv


def get_env_info_single(env):
    obs = env.reset()

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
    o, r, d, i = env.step(action)

    print("State Length: ", obs['agent0']['state'].shape)
    print("Image Shape: ", obs['agent0']['image'][0, 0, :, 0])
    print("Env Info: ", i['agent0'])


if __name__ == '__main__':
    # env = MultiAgentsEnv(num_agents=8)
    env = SingleAgentsEnv()
    get_env_info_single(env)
