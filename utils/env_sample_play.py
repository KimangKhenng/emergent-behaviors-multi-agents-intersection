from envs.multi_agents import MultiAgentsEnv

env = MultiAgentsEnv(num_agents=8)
obs = env.reset()
frames = []

if __name__ == '__main__':
    d = {}
    d["__all__"] = False
    for t in range(1000):
        if d["__all__"]:
            # frames.append(frame)
            continue
        action = env.action_space.sample()
        # print("-------------------")
        # print("Action Before: ", action)
        for a in action.values():
            a[-1] = 1.0
        # print("-------------------")
        # print("Action After: ", action)
        o, r, d, i = env.step(action)
        # print(r)
        print("step: ", t)
        frame = env.render(mode="top_down", film_size=(1000, 1000), track_target_vehicle=False,
                           screen_size=(1000, 1000))

        # frames.append(frame)
    env.close()
