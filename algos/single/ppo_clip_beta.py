import math
from collections import OrderedDict
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from GPUtil import showUtilization as gpu_usage
import gc

from model.actor_critic import ActorCritics

# ## Split the GPU memory allocation
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'garbage_collection_threshold:0.6'
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

################################## set device ##################################
print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


def get_joined_action_state(actions, state):
    joined_states = OrderedDict()
    for key in state.keys():
        joined_states[key] = state[key]['state']
    arrays = [v for k, v in actions.items()]  # extract arrays using a list comprehension
    concat_actions = np.concatenate(arrays)  # concatenate arrays into a single array
    arrays = [v for k, v in joined_states.items()]
    concat_states = np.concatenate(arrays)
    joined_actions_states = np.concatenate((concat_actions, concat_states))
    joined_actions_states = torch.from_numpy(joined_actions_states)
    # print("joined_actions_states.shape: ", joined_actions_states.shape)
    if joined_actions_states.shape[0] < 84:
        padding = (0, 84 - joined_actions_states.shape[0])
        joined_actions_states = torch.nn.functional.pad(joined_actions_states, padding)
    return joined_actions_states


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.next_states = []
        self.front_views = []
        self.next_front_views = []
        self.rewards = []
        self.logprobs = []
        self.state_action_values = []
        self.is_terminated = []

    def add_reward(self, reward):
        self.rewards.append(reward)

    def add_terminated(self, terminated):
        self.is_terminated.append(terminated)

    def add_actions(self, actions):
        self.actions.append(actions)

    def add_state(self, state):
        self.states.append(state)

    def add_front_view(self, front_view):
        self.front_views.extend(front_view)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.front_views[:]
        del self.rewards[:]
        del self.state_action_values[:]
        del self.is_terminated[:]
        del self.logprobs[:]
        del self.next_states[:]


class SinglePPOClipBetaAgent(nn.Module):
    def __init__(
            self,
            state_size=21,
            batch_size=64,
            lr_actor=0.1,
            lr_critic=0.1,
            gamma=0.99,
            k_epochs=10,
            eps_clip=0.2):
        # Initialize the network and optimizer
        super().__init__()
        # Central Policy for learning
        self.K_epochs = k_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.policy = ActorCritics(state_size=state_size).to(device)
        # Optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritics(state_size=state_size).to(device)

        self.policy_old.load_state_dict(self.policy.state_dict())

        # Mean Squared Error Loss
        self.MseLoss = nn.MSELoss()

        # Rollout Buffer
        self.rollout_buffer = RolloutBuffer()

    # Select individual action
    def select_action(self, obs):
        # Assume that state is for individual agent with data type of dict with keys 'state' and 'image'
        # print("state: ", obs['state'].shape)
        rgb_camera = obs['image']
        # print("rgb_camera: ", rgb_camera.shape)
        with torch.no_grad():
            state = torch.Tensor(obs['state']).to(device)
            front_view = torch.Tensor(rgb_camera).permute(2, 0, 1).unsqueeze(0).to(device)
            action, logprobs, state_action_val = self.policy_old.act(
                state=state,
                front_view=front_view,
            )
            # print("action: ", action)
            # state_action = numpy.concatenate((obs['state'], action))
            # state_action_value = self.policy.critic(torch.FloatTensor(state_action).to(device))

        self.rollout_buffer.states.append(state)
        self.rollout_buffer.front_views.append(front_view)
        self.rollout_buffer.add_actions(torch.Tensor(action))
        self.rollout_buffer.logprobs.append(torch.Tensor(logprobs))
        self.rollout_buffer.state_action_values.append(state_action_val)

        return action

    def update(self):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        # print("Reward Length")
        # print(len(self.rollout_buffer.rewards))
        # print("Termination Length")
        # print(len(self.rollout_buffer.is_terminated))
        for reward, is_terminated in zip(reversed(self.rollout_buffer.rewards),
                                         reversed(self.rollout_buffer.is_terminated)):
            if is_terminated:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        # rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # print("State: ", self.rollout_buffer.states)
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.rollout_buffer.states, dim=0)).detach()
        old_front_view = torch.squeeze(torch.stack(self.rollout_buffer.front_views, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.rollout_buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.rollout_buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.rollout_buffer.state_action_values, dim=0)).detach().to(
            device)
        # old_states = torch.squeeze(torch.stack(self.rollout_buffer.states, dim=0)).detach()
        # old_front_view = torch.squeeze(torch.stack(self.rollout_buffer.front_views, dim=0)).detach()
        # old_actions = torch.squeeze(torch.stack(self.rollout_buffer.actions, dim=0)).detach()
        # old_logprobs = torch.squeeze(torch.stack(self.rollout_buffer.logprobs, dim=0)).detach()
        # old_state_values = torch.squeeze(torch.stack(self.rollout_buffer.state_action_values, dim=0)).detach()

        print("Old State Values: ", old_state_values.shape)
        print("Rewards: ", rewards.shape)
        print("Old Frontview: ", old_front_view.shape)
        print("old_actions: ", old_actions.shape)
        # print("Old Value state: ", old_state_values.shape)
        # calculate advantages

        # optim_iter_num = int(math.ceil(old_states.shape[0] / self.optim_batch_size))
        # print("Optim Iter Num: ", optim_iter_num)
        # Optimize policy for K epochs
        # Use torch.utils.data to create a DataLoader
        # that will take care of creating batches
        dataset = TensorDataset(old_states, old_front_view, old_actions, old_logprobs, old_state_values, rewards)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        for step in range(self.K_epochs):
            for id_batch, (old_states_batch,
                           old_front_view_batch,
                           old_actions_batch,
                           old_logprobs_batch,
                           old_state_values_batch,
                           rewards_batch
                           ) in enumerate(dataloader):
                print("Performing Optimization: ", step)
                print("Batch ID: ", id_batch)
                # old_states_batch_gpu = old_states_batch.to(device)
                # old_front_view_batch_gpu = old_front_view_batch.to(device)
                # old_actions_batch_gpu = old_actions_batch.to(device)
                # old_state_values_batch_gpu = old_state_values_batch.to(device)
                # rewards_batch_gpu = rewards_batch.to(device)
                # old_logprobs_batch_gpu = old_logprobs_batch.to(device)
                logprobs, dist_entropy, action_state_value = self.policy.evaluate(
                    old_states_batch,
                    old_front_view_batch,
                    old_actions_batch,
                )
                # print("action_state_value: ", action_state_value.shape)

                # match state_values tensor dimensions with rewards tensor
                action_state_value = torch.squeeze(action_state_value)
                advantages = rewards_batch.detach() - old_state_values_batch.detach()
                # Finding the ratio (pi_theta / pi_theta__old)
                # print("logprobs: ", logprobs.shape)
                # print("old_logprobs: ", old_logprobs.shape)
                ratios = torch.exp(logprobs - old_logprobs_batch.detach())

                # Finding Surrogate Loss

                # print("ratios: ", ratios.shape)
                advantages = advantages.unsqueeze(1)
                # print("advantages: ", advantages.shape)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(action_state_value,
                                                                     rewards_batch) - 0.01 * dist_entropy

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                advantages = advantages.squeeze(1)

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.rollout_buffer.clear()
        gc.collect()
        torch.cuda.empty_cache()
        del loss
        gpu_usage()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
