from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.beta import Beta

from model.central_actor_critics import CentralActorCritics
from model.indi_actor_critics import IndividualActorCritics

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
    joined_actions_states = torch.from_numpy(joined_actions_states)
    if joined_actions_states.shape[0] < 168:
        padding = (0, 168 - joined_actions_states.shape[0])
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
        self.joined_action_state = []
        self.is_terminated = []

    def add_next_state_action_values(self, actions, state):
        joined_actions_states = get_joined_action_state(actions, state)
        self.joined_action_state.append(joined_actions_states)

    def add_reward(self, rewards_dict, num_agent):
        # if len(rewards_dict) != num_agent:
        #     all_agent_names = [f'agent{i}' for i in range(num_agent)]
        #     for agent_name in all_agent_names:
        #         if agent_name not in rewards_dict:
        #             rewards_dict[agent_name] = -5
        # Extract the reward values and maintain chronological order
        # rewards_list = [rewards_dict[f'agent{i}'] for i in range(len(rewards_dict))]
        rewards_list = [i for i in rewards_dict.values()]
        self.rewards.extend(rewards_list)

    def add_terminated(self, terminated_dict, num_agent):
        # if len(terminated_dict) - 1 != num_agent:
        #     all_agent_names = [f'agent{i}' for i in range(num_agent)]
        #     for agent_name in all_agent_names:
        #         if agent_name not in terminated_dict:
        #             terminated_dict[agent_name] = True
        # Extract the done values and maintain chronological order
        # print(terminated_dict)
        # terminated_list = [terminated_dict[f'agent{i}'] for i in range(len(terminated_dict) - 1)]
        terminated_list = [i for i in terminated_dict.values()]
        self.is_terminated.extend(terminated_list)

    # def add_state(self, state_dict, num_agent):
    #     print(state_dict)
    #     if len(state_dict) != num_agent:
    #         all_agent_names = [f'agent{i}' for i in range(num_agent)]
    #         for agent_name in all_agent_names:
    #             if agent_name not in state_dict:
    #                 state_dict[agent_name] = np.zeros(19)
    #     # Extract the reward values and maintain chronological order
    #     state_list = [state_dict[f'agent{i}'] for i in range(len(state_dict))]
    #     self.states.append(state_list)

    def add_actions(self, actions):
        self.actions.extend(actions)

    def add_state(self, state):
        self.states.append(state)

    def add_front_view(self, front_view):
        self.front_views.extend(front_view)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.rewards[:]
        del self.state_action_values[:]
        del self.joined_action_state[:]
        del self.is_terminated[:]
        del self.logprobs[:]
        del self.next_states[:]


class CentralPPOAgents(nn.Module):
    def __init__(
            self,
            num_agents=8,
            state_size=19,
            joined_actions_size=0,
            lr_actor=0.1,
            lr_critic=0.1,
            gamma=0.99,
            k_epochs=80,
            eps_clip=0.2):
        # Initialize the network and optimizer
        super().__init__()
        # Central Policy for learning
        self.num_agents = num_agents
        self.K_epochs = k_epochs
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.central_policy = CentralActorCritics(state_size=state_size, num_agents=num_agents)
        # Optimizer
        self.optimizer = torch.optim.Adam([
            {'params': self.central_policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.central_policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = CentralActorCritics(state_size=state_size, num_agents=num_agents)

        # Decentralized Policies
        self.decentralized_policies = OrderedDict(
            ('agent{}'.format(i), IndividualActorCritics(state_size=state_size)) for i in range(num_agents))

        for key in self.decentralized_policies.keys():
            self.policy_old.actor.load_state_dict(self.decentralized_policies[key].actor.state_dict())

        # Mean Squared Error Loss
        self.MseLoss = nn.MSELoss()

        # Rollout Buffer
        self.rollout_buffer = RolloutBuffer()

    # Select individual action
    def select_action(self, agent_name, states):
        # Assume that state is for individual agent with data type of dict with keys 'state' and 'image'
        with torch.no_grad():
            state = torch.FloatTensor(states['state'])
            dept_camera = states['image'][:, :, :, -1]
            front_view = torch.FloatTensor(dept_camera).permute(2, 0, 1)
            action, logprobs = self.decentralized_policies[agent_name].act(
                state=state,
                front_view=front_view,
            )
        self.rollout_buffer.states.append(state)
        self.rollout_buffer.front_views.append(front_view)
        return action, logprobs

    def select_actions(self, states):
        # Assume that the state is the dictionary of states for all agents
        actions = OrderedDict()
        logprobs = OrderedDict()
        joined_states = OrderedDict()
        # Loop through all agents by key
        for key in states.keys():
            # Select action for each agent
            action, logprob = self.select_action(key, states[key])
            actions[key] = action
            logprobs[key] = logprob
            joined_states[key] = states[key]['state']
            # state roll out buffer
            self.rollout_buffer.add_actions(torch.FloatTensor(action))
            self.rollout_buffer.logprobs.append(torch.Tensor(logprob))

        joined_actions_states = get_joined_action_state(actions, states)
        with torch.no_grad():
            state_action_values = self.central_policy.critic(joined_actions_states)

        # print("state action values: ", state_action_values)
        # Append to the rollout buffer

        for _ in states.keys():
            self.rollout_buffer.state_action_values.append(state_action_values)
            self.rollout_buffer.joined_action_state.append(joined_actions_states)
        return actions

    def update(self):
        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminated in zip(reversed(self.rollout_buffer.rewards),
                                         reversed(self.rollout_buffer.is_terminated)):
            if is_terminated:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # print("State: ", self.rollout_buffer.states)
        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.rollout_buffer.states, dim=0)).detach()
        old_front_view = torch.squeeze(torch.stack(self.rollout_buffer.front_views, dim=0)).detach()
        old_actions = torch.squeeze(torch.stack(self.rollout_buffer.actions, dim=0)).detach()
        old_logprobs = torch.squeeze(torch.stack(self.rollout_buffer.logprobs, dim=0)).detach()
        old_joined_actions_states = torch.squeeze(
            torch.stack(self.rollout_buffer.joined_action_state, dim=0)).detach()
        old_state_values = torch.squeeze(torch.stack(self.rollout_buffer.state_action_values, dim=0)).detach()

        print("Old State Values: ", old_state_values.shape)
        print("Rewards: ", rewards.shape)
        print("Old Frontview: ", old_front_view.shape)
        print("old_actions: ", old_actions.shape)
        print("old_joined_actions_states: ", old_joined_actions_states.shape)
        # print("Old Value state: ", old_state_values.shape)
        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(5):
            # Evaluating old actions and values
            # print("Old Front View: ", old_front_view.shape)
            logprobs, dist_entropy, action_state_value = self.central_policy.evaluate(
                old_states,
                old_front_view,
                old_actions,
                old_joined_actions_states
            )

            # match state_values tensor dimensions with rewards tensor
            action_state_value = torch.squeeze(action_state_value)

            # Finding the ratio (pi_theta / pi_theta__old)
            # print("logprobs: ", logprobs.shape)
            # print("old_logprobs: ", old_logprobs.shape)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss

            # print("ratios: ", ratios.shape)
            advantages = advantages.unsqueeze(1)
            print("advantages: ", advantages.shape)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # print("surr1: ", surr1.shape)
            # print("surr2: ", surr2.shape)
            # print("action_state_value: ", action_state_value.shape)
            # print("rewards: ", rewards.shape)
            # print("dist_entropy: ", dist_entropy)
            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(action_state_value, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

            advantages = advantages.squeeze(1)

        # Copy new weights into old policy
        for policy in self.decentralized_policies.values():
            self.policy_old.actor.load_state_dict(policy.actor.state_dict())

        # clear buffer
        self.rollout_buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
