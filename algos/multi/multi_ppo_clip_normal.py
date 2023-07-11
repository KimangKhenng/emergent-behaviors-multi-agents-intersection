import itertools
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from algos.multi.buffer import Buffer
from algos.utils.reward import meanfield_approximation
from model.network import JoinedGaussianPolicy

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


def get_available_agents(rewards, dones):
    # Check the number of agents available in both reward and done
    available_agents = set(rewards[0].keys()).intersection(set(dones[0].keys()))
    return list(available_agents)


def get_agent_info(agent_name, rewards, dones):
    # Check the number of agents available only if each agent is available in both reward and done
    available_agents = set(rewards[0].keys()).intersection(set(dones[0].keys()))
    if agent_name not in available_agents:
        print(f"Agent '{agent_name}' is not available in the data")
        return None

    # Extract the reward and done condition for the specified agent
    reward_agent = []
    done_agent = []
    for t in range(len(rewards)):
        if agent_name in rewards[t] and agent_name in dones[t]:
            reward_agent.append(rewards[t][agent_name])
            done_agent.append(dones[t][agent_name])
        else:
            print(f"Missing data for agent '{agent_name}' at time step {t}")
            return None

    return reward_agent, done_agent


def get_joined_action_state(actions, state):
    joined_states = OrderedDict()
    for key in state.keys():
        joined_states[key] = state[key]['state']
    arrays = [v for k, v in actions.items()]  # extract arrays using a list comprehension
    concat_actions = np.concatenate(arrays)  # concatenate arrays into a single array
    arrays = [v for k, v in joined_states.items()]
    concat_states = np.concatenate(arrays)
    joined_actions_states = np.concatenate((concat_actions, concat_states))
    joined_actions_states = torch.from_numpy(joined_actions_states).to(device)
    # print("joined_actions_states.shape: ", joined_actions_states.shape)
    # if joined_actions_states.shape[0] < 84:
    #     padding = (0, 84 - joined_actions_states.shape[0])
    #     joined_actions_states = torch.nn.functional.pad(joined_actions_states, padding)
    return joined_actions_states


class MultiPPOClipNormalAgents(nn.Module):
    def __init__(
            self,
            num_agents,
            batch_size,
            state_dim,
            action_dim,
            lr_actor,
            lr_critic,
            gamma,
            K_epochs,
            eps_clip,
            hidden_dim,
            action_std_init=0.6
    ):
        # Initialize the network and optimizer
        super().__init__()
        # Central Policy for learning
        self.num_agents = num_agents
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.central_policy = JoinedGaussianPolicy(input_size=state_dim,
                                                   action_size=action_dim,
                                                   hidden_size=hidden_dim,
                                                   action_std_init=action_std_init).to(device)
        self.action_std = action_std_init
        # Optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.central_policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.central_policy.critic.parameters(), 'lr': lr_critic}
        ])

        # Decentralized Policies
        self.decentralized_policies = OrderedDict(
            ('agent{}'.format(i),
             JoinedGaussianPolicy(input_size=state_dim, action_size=action_dim, hidden_size=hidden_dim,
                                  action_std_init=action_std_init).to(device)) for i in
            range(num_agents))

        for key in self.decentralized_policies.keys():
            self.decentralized_policies[key].load_state_dict(self.central_policy.state_dict())

        # Mean Squared Error Loss
        self.MseLoss = nn.MSELoss()

        self.rollout_buffer = Buffer()

    def set_action_std(self, new_action_std):
        self.action_std = new_action_std
        self.central_policy.set_action_std(new_action_std)
        for key in self.decentralized_policies.keys():
            self.decentralized_policies[key].set_action_std(new_action_std)

    def decay_action_std(self, action_std_decay_rate, min_action_std):
        self.action_std = self.action_std - action_std_decay_rate
        self.action_std = round(self.action_std, 4)
        if self.action_std <= min_action_std:
            self.action_std = min_action_std
            print("setting actor output action_std to min_action_std : ", self.action_std)
        else:
            print("setting actor output action_std to : ", self.action_std)
        self.set_action_std(self.action_std)

    # Select individual action
    def select_action(self, agent_name, states):
        with torch.no_grad():
            state = torch.FloatTensor(states).to(device)
            action, logprobs = self.decentralized_policies[agent_name].act(
                state=state,
            )
        return action, logprobs

    def select_actions(self, states):
        # Assume that the state is the dictionary of states for all agents
        joined_action = []
        actions = OrderedDict()
        logprobs = OrderedDict()
        states_values = OrderedDict()
        state_action = OrderedDict()
        for key in states.keys():
            action, logprob = self.select_action(key, states[key])
            joined_action.append(action)
            actions[key] = action
            logprobs[key] = logprob
            # states_values[key] = state_val

        for key in states.keys():
            # contact states[key] and actions
            joined_action = np.array(joined_action)
            joined_action = joined_action.reshape(-1)
            meanfield_action = meanfield_approximation(joined_action)
            # if joined_action.size < 8:
            #     # Calculate the amount of padding needed
            #     pad_width = 8 - joined_action.size
            #
            #     # Pad the array with zeros
            #     joined_action = np.pad(joined_action, (0, pad_width), mode='constant')

            joined_state_action = np.append(states[key], meanfield_action)
            joined_state_action = joined_state_action.astype(np.float32)
            state_action[key] = joined_state_action
            state_action_value = self.decentralized_policies[key].critic(
                torch.tensor(joined_state_action).to(device))
            states_values[key] = state_action_value

        self.rollout_buffer.states.append(states)
        self.rollout_buffer.actions.append(actions)
        self.rollout_buffer.joined_actions_states.append(state_action)
        self.rollout_buffer.logprobs.append(logprobs)
        self.rollout_buffer.state_value.append(states_values)
        return actions

    def update(self):
        rewards = []
        discounted_reward = 0
        self.rollout_buffer.rearrange()
        for reward, is_terminated in zip(reversed(self.rollout_buffer.rewards),
                                         reversed(self.rollout_buffer.is_terminated)):
            if is_terminated:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        old_states = torch.tensor(self.rollout_buffer.states).detach().to(device)
        old_actions = torch.tensor(self.rollout_buffer.actions).detach().to(device)
        old_joined_actions_states = torch.tensor(self.rollout_buffer.joined_actions_states).detach().to(device)
        old_logprobs = torch.tensor(self.rollout_buffer.logprobs).detach().to(device)
        old_state_values = torch.tensor(self.rollout_buffer.state_value).detach().to(
            device)

        dataset = TensorDataset(old_states, old_actions, old_joined_actions_states, old_logprobs, old_state_values,
                                rewards)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Optimize policy for K epochs
        for step in range(self.K_epochs):
            # print("Performing Optimization: ", step)
            # print("Slicing Batch")
            for id_batch, (old_states_batch,
                           old_actions_batch,
                           old_joined_actions_states_batch,
                           old_logprobs_batch,
                           old_state_values_batch,
                           rewards_batch,
                           ) in enumerate(dataloader):
                # print("Batch: ", id_batch)
                logprobs, dist_entropy, action_state_value = self.central_policy.evaluate(
                    old_states_batch,
                    old_actions_batch,
                    old_joined_actions_states_batch
                )

                # match state_values tensor dimensions with rewards tensor
                action_state_value = torch.squeeze(action_state_value)
                advantages = rewards_batch.detach() + action_state_value.detach() - old_state_values_batch.detach()

                ratios = torch.exp(logprobs - old_logprobs_batch.detach())

                # Finding Surrogate Loss

                advantages = advantages.unsqueeze(1)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(action_state_value,
                                                                     rewards_batch) - 0.01 * dist_entropy

                # take gradient step
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                # advantages = advantages.squeeze(1)

        # Copy new weights into old policy
        for key in self.decentralized_policies.keys():
            self.decentralized_policies[key].load_state_dict(self.central_policy.state_dict())

        # clear buffer
        self.rollout_buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.central_policy.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.central_policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        for key in self.decentralized_policies.keys():
            self.decentralized_policies[key].load_state_dict(
                torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
