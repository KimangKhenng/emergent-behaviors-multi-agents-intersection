from metadrive import MultiAgentIntersectionEnv
from env.mock import MultiAgentEnv
import numpy as np
import torch
from torchsummary import summary
from torchviz import make_dot
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from model.network import Policy

NUM_AGENTS = 4

env = MultiAgentIntersectionEnv(config={
    "num_agents": NUM_AGENTS,
    "use_render": False,
    "manual_control": False,
    "map_config": {
        "lane_num": 1
    },
    "vehicle_config": {
        "show_lidar": True,
        "image_source": "rgb_camera",
        "rgb_camera": (84, 84)
    },
    "show_fps": True,
    "image_observation": True,
})
obs = env.reset()
#
# mock = MultiAgentEnv(NUM_AGENTS)
#
# obs_mock = mock.reset()

# print("Mock", len(obs_mock['agent1']))
# print("Mock Action", mock.action_space.sample())

# obs_size = len(obs['agent0'])
# action_size = env.action_space.sample()
#
# # print("Real", obs_size)
# print("Real Reward", action_size)

print("Sate Length: ", obs['agent0']['state'].shape)
print("Image Shape: ", obs['agent0']['image'].shape)

state = torch.from_numpy(obs['agent0']['state'])
sample_image = obs['agent0']['image']
torch_tensor = torch.from_numpy(sample_image)
torch_tensor = torch_tensor.permute(3, 2, 0, 1)

print("Image Shape: ", torch_tensor.shape)

model = Policy(state_size=19, image_width=84, image_height=84, action_size=2)

result = model.forward(x_img=torch_tensor, x_vec=state)
print(result)

summary(model, [torch_tensor, state])
# Define the hyperparameters
# learning_rate = 0.01
# gamma = 0.99
# num_episodes = 1000
#
# # Initialize the policy and action arrays for each agent
# policies = []
# actions = []
#
# for i in range(env.num_agents):
#     # policies.append(np.random.rand(env.observation_space['agent{}'.format(i)].shape[0],
#     #                                env.action_space['agent{}'.format(i)].shape[0]))
#     # actions.append(np.zeros(env.action_space['agent{}'.format(i)].shape[0]))
#     policy = Policy(env.observation_space['agent{}'.format(i)].shape[0], env.action_space['agent{}'.format(i)].shape[0])
#     optimizer = optim.Adam(policy.parameters(), lr=learning_rate)
#     policies.append(policy)
#     actions.append(np.zeros(env.action_space['agent{}'.format(i)].shape[0]))
#
#
# # Define the PG update function
# def pg_update(episode_states, episode_actions, episode_rewards, policy):
#     # Convert the episode data to PyTorch tensors
#     episode_states = torch.tensor(episode_states, dtype=torch.float)
#     episode_actions = torch.tensor(episode_actions, dtype=torch.float)
#     episode_rewards = torch.tensor(episode_rewards, dtype=torch.float)
#
#     # Calculate the discounted rewards-to-go
#     episode_returns = torch.zeros_like(episode_rewards)
#     running_total = 0
#     for t in reversed(range(len(episode_rewards))):
#         running_total = running_total * gamma + episode_rewards[t]
#         episode_returns[t] = running_total
#
#     # Normalize the rewards
#     episode_returns -= torch.mean(episode_returns)
#     episode_returns /= torch.std(episode_returns)
#
#     # Calculate the loss and update the policy
#     optimizer.zero_grad()
#     loss = -torch.mean(torch.sum(torch.log(policy(episode_states)) * episode_actions, dim=1) * episode_returns)
#     loss.backward()
#     optimizer.step()
#
#
# # Train the agents using the PG algorithm
# def find_missing_key(dict1, dict2):
#     for key in dict1:
#         if key not in dict2:
#             return key
#     return None
#
#
# for episode in range(num_episodes):
#     # Reset the environment and get the initial state
#     state = env.reset()
#
#     # Collect data for each agent in the episode
#     episode_states = [[] for _ in range(env.num_agents)]
#     episode_actions = [[] for _ in range(env.num_agents)]
#     episode_rewards = [[] for _ in range(env.num_agents)]
#     total_reward = 0
#     # done = False
#     d = {}
#     d["__all__"] = False
#     while not d['__all__']:
#         # Get the actions for each agent from their policy
#         final_actions = {}
#         # for i in range(env.num_agents):
#         #     if 'agent{}'.format(i) in reward_dict:
#         #         state_tensor = torch.tensor(state['agent{}'.format(i)], dtype=torch.float)
#         #         action_tensor = policies[i](state_tensor)
#         #         # actions[i] = action_tensor.detach().numpy()
#         #         final_actions['agent{}'.format(i)] = action_tensor.detach().numpy()
#         for index, key in enumerate(state):
#             state_tensor = torch.tensor(state[key], dtype=torch.float)
#             action_tensor = policies[index](state_tensor)
#             actions[index] = action_tensor.detach().numpy()
#             final_actions[key] = actions[index]
#             # Step the environment with the actions
#             # final_actions = OrderedDict([('agent{}'.format(i), actions[i]) for i in range(env.num_agents)])
#
#         # print(final_actions)
#         # for a in final_actions.values():
#         #     a[-1] = 1.0
#         # print(final_actions)
#         next_state, reward_dict, d, info_dict = env.step(final_actions)
#         # print(reward_dict)
#         # env.render(mode="top_down", film_size=(500, 500), track_target_vehicle=False, screen_size=(500, 500))
#
#         # Collect the data for each agent
#         for i in range(env.num_agents):
#             if 'agent{}'.format(i) in reward_dict:
#                 episode_states[i].append(state['agent{}'.format(i)])
#                 episode_actions[i].append(actions[i])
#                 # print(reward_dict)
#                 # print('agent{}'.format(i), reward_dict['agent{}'.format(i)])
#                 episode_rewards[i].append(reward_dict['agent{}'.format(i)])
#                 total_reward += reward_dict['agent{}'.format(i)]
#
#         # Update the current state
#         state = next_state
#
#     # Update the policies for each agent
#     for i in range(env.num_agents):
#         if episode_rewards[i]:
#             pg_update(episode_states[i], episode_actions[i], episode_rewards[i], policies[i])
#
#     # Print the episode number and total reward
#     print("Episode {} Total Reward: {}".format(episode, total_reward))
#
# frames = []
#
# d = {}
# d["__all__"] = False
# for i in range(1000):
#     if d["__all__"]:
#         #frames.append(frame)
#         continue
#     action = env.action_space.sample()
#     for a in action.values():
#         a[-1] = 1.0
#     o, r, d, i = env.step(action)
#     # print(r)
#     frame = env.render(mode="top_down", film_size=(1000, 1000), track_target_vehicle=False, screen_size=(1000, 1000))
#     # frames.append(frame)
# env.close()
#
# # render image
# print("\nGenerate gif...")
# imgs = [pygame.surfarray.array3d(frame) for frame in frames]
# imgs = [Image.fromarray(img) for img in imgs]
# imgs[0].save("demo.gif", save_all=True, append_images=imgs[1:], duration=50, loop=0)
