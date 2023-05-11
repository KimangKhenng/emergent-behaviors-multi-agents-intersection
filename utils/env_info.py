from envs.multi_agents import MultiAgentsEnv

if __name__ == '__main__':
    env = MultiAgentsEnv(num_agents=8)
    obs = env.reset()

    action = env.action_space.sample()
    for a in action.values():
        a[-1] = 1.0
    o, r, d, i = env.step(action)

    print("State Length: ", obs['agent0']['state'].shape)
    print("Image Shape: ", obs['agent0']['image'][0, 0, :, 0])
    print("Env Info: ", i['agent0'])
