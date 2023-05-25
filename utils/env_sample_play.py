from envs.multi_agents import MultiAgentsEnv
from envs.single_agent_intersection import SingleAgentsEnv
from metadrive.examples import expert
# env = MultiAgentsEnv(num_agents=8)
env = SingleAgentsEnv()
obs = env.reset()
frames = []
ep_reward = 0.0
if __name__ == '__main__':
    d = {}
    d["__all__"] = False
    for t in range(1000):
        # if d["__all__"]:
        #     # frames.append(frame)
        #     continue
        # action = env.action_space.sample()
        # print("-------------------")
        # print("-------------------")
        # print("Action Before: ", action)
        # for a in action.values():
        #     a[-1] = 1.0
        # print("-------------------")
        # print("Action After: ", action)
        # print("Action: ", expert(env.vehicle))
        o, r, d, i = env.step(expert(env.vehicle))
        print("reward: ", r)
        print("done: ", d)
        ep_reward += r
        # print(r)
        print("step: ", t)
        env.render(mode="top_down", film_size=(1000, 1000), track_target_vehicle=True,
                   screen_size=(1000, 1000))
        if d:
            print("Arriving Destination: {}".format(i["arrive_dest"]))
            print("\nEpisode reward: ", ep_reward)
            break

        # frames.append(frame)
    env.close()
