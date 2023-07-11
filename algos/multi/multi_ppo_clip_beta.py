import itertools
import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from algos.multi.buffer import Buffer
from algos.utils.reward import monte_carlo_state_rewards, generalized_advantage_estimation, meanfield_approximation
from model.network import JoinedBetaPolicy

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


class MultiPPOClipBetaAgents(nn.Module):
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
            hidden_dim
    ):
        # Initialize the network and optimizer
        super().__init__()
        # Central Policy for learning
        self.num_agents = num_agents
        self.K_epochs = K_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.lambda_ = 0.95
        self.central_policy = JoinedBetaPolicy(input_size=state_dim, action_size=action_dim, hidden_size=hidden_dim).to(
            device)
        # Optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.central_policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.central_policy.critic.parameters(), 'lr': lr_critic}
        ])

        # self.policy_old = CentralActorCritics(state_size=state_size, num_agents=num_agents).to(device)

        # Decentralized Policies
        self.decentralized_policies = OrderedDict(
            ('agent{}'.format(i),
             JoinedBetaPolicy(input_size=state_dim, action_size=action_dim, hidden_size=hidden_dim).to(device)) for i in
            range(num_agents))

        for key in self.decentralized_policies.keys():
            self.decentralized_policies[key].load_state_dict(self.central_policy.state_dict())

        # Mean Squared Error Loss
        self.MseLoss = nn.MSELoss()

        # Rollout Buffer
        self.rollout_buffer = Buffer()

    # Select individual action
    def select_action(self, agent_name, states):
        # Assume that state is for individual agent with data type of dict with keys 'state' and 'image'
        with torch.no_grad():
            state = torch.FloatTensor(states).to(device)
            # rgb_camera = states['image']
            # front_view = torch.FloatTensor(rgb_camera).permute(2, 0, 1).unsqueeze(0).to(device)
            action, logprobs = self.decentralized_policies[agent_name].act(
                state=state,
            )
            # action = [np.float32(item) for item in action]
            # print(action, type(action[0]))
            # print(mapped_action, type(mapped_action[0]))
        # self.buffer.states.append(state)
        # self.rollout_buffer.front_views.append(front_view)
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
        self.rollout_buffer.rearrange()
        # Monte Carlo estimate of state rewards:
        # rewards = monte_carlo_state_rewards_v0(rewards=self.rollout_buffer.rewards,
        #                                        termination_states=self.rollout_buffer.is_terminated,
        #                                        gamma=self.gamma)
        rewards = generalized_advantage_estimation(rewards=self.rollout_buffer.rewards,
                                                   termination_states=self.rollout_buffer.is_terminated,
                                                   gamma=self.gamma)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # print("State: ", self.rollout_buffer.states)
        # convert list to tensor
        old_states = torch.tensor(self.rollout_buffer.states).detach().to(device)
        old_actions = torch.tensor(self.rollout_buffer.actions).detach().to(device)
        old_joined_actions_states = torch.tensor(self.rollout_buffer.joined_actions_states).detach().to(device)
        old_logprobs = torch.tensor(self.rollout_buffer.logprobs).detach().to(device)
        # old_joined_actions_states = torch.squeeze(
        #     torch.stack(self.buffer.joined_action_state, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.rollout_buffer.state_value, dim=0)).detach().to(device)

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
                # Evaluating old actions and values
                # print("Old Front View: ", old_front_view.shape)
                # print("Old Actions Batched: ", old_actions_batch.shape)
                # print("Old Front View Batched: ", old_front_view_batch.shape)
                # print("Old States Batched: ", old_states_batch.shape)
                # print("Old Joined Actions States Batched: ", old_joined_actions_states_batch.shape)
                logprobs, dist_entropy, state_value = self.central_policy.evaluate(
                    old_states_batch,
                    old_actions_batch,
                    old_joined_actions_states_batch
                )
                # print("action_state_value: ", action_state_value.shape)

                # match state_values tensor dimensions with rewards tensor
                state_value = torch.squeeze(state_value)
                # print("rewards_batch: ", rewards_batch.shape)
                # print("state_value: ", old_state_values_batch.shape)
                advantages = rewards_batch.detach() + state_value.detach() - old_state_values_batch.detach()

                # Finding the ratio (pi_theta / pi_theta__old)
                # print("logprobs: ", logprobs.shape)
                # print("old_logprobs: ", old_logprobs.shape)
                ratios = torch.exp(logprobs - old_logprobs_batch.detach())
                # print("Ratios: ", ratios.shape)

                # Finding Surrogate Loss

                # print("ratios: ", ratios.shape)
                # advantages = advantages.unsqueeze(1)
                # print("advantages: ", advantages.shape)
                advantages = advantages.unsqueeze(1)
                # print("Advantages: ", advantages.shape)
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # print("surr1: ", surr1.shape)
                # print("surr2: ", surr2.shape)
                # print("action_state_value: ", action_state_value.shape)
                # print("rewards: ", rewards.shape)
                # print("dist_entropy: ", dist_entropy)
                # final loss of clipped objective PPO
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_value,
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
