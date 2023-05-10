from envs.multi_agents import MultiAgentsEnv

if __name__ == '__main__':
    env = MultiAgentsEnv(num_agents=8)
    obs = env.reset()

    print("State Length: ", obs['agent0']['state'].shape)
    print("Image Shape: ", obs['agent0']['image'][0, 0, :, 0])
