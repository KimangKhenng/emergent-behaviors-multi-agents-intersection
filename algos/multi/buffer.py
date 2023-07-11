import itertools

import numpy as np


class Buffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.joined_actions_states = []
        self.state_value = []
        self.rewards = []
        self.logprobs = []
        self.is_terminated = []

    def rearrange(self):
        for item in self.is_terminated:
            item.pop('__all__', None)
        sequential_states = {}
        sequential_actions = {}
        sequential_joined_actions_states = {}
        sequential_state_value = {}
        sequential_rewards = {}
        sequential_logprobs = {}
        sequential_is_terminated = {}
        for state, action, joined_actions_states, state_value, rewards, logprobs, is_terminated in zip(self.states,
                                                                                                       self.actions,
                                                                                                       self.joined_actions_states,
                                                                                                       self.state_value,
                                                                                                       self.rewards,
                                                                                                       self.logprobs,
                                                                                                       self.is_terminated
                                                                                                       ):
            for agent, value in state.items():
                if agent not in sequential_states:
                    sequential_states[agent] = []
                sequential_states[agent].append(value)
            for agent, value in action.items():
                if agent not in sequential_actions:
                    sequential_actions[agent] = []
                sequential_actions[agent].append(value)
            for agent, value in joined_actions_states.items():
                if agent not in sequential_joined_actions_states:
                    sequential_joined_actions_states[agent] = []
                sequential_joined_actions_states[agent].append(value)
            for agent, value in state_value.items():
                if agent not in sequential_state_value:
                    sequential_state_value[agent] = []
                sequential_state_value[agent].append(value)
            for agent, value in rewards.items():
                if agent not in sequential_rewards:
                    sequential_rewards[agent] = []
                sequential_rewards[agent].append(value)
            for agent, value in logprobs.items():
                if agent not in sequential_logprobs:
                    sequential_logprobs[agent] = []
                sequential_logprobs[agent].append(value)
            for agent, value in is_terminated.items():
                if agent not in sequential_is_terminated:
                    sequential_is_terminated[agent] = []
                sequential_is_terminated[agent].append(value)
        # Transform the dictionary into a list
        self.states = list(sequential_states.values())
        self.states = list(itertools.chain.from_iterable(self.states))
        # self.states = np.array(self.states)

        self.actions = list(sequential_actions.values())
        self.actions = list(itertools.chain.from_iterable(self.actions))
        # self.actions = np.array(self.actions)

        self.joined_actions_states = list(sequential_joined_actions_states.values())
        self.joined_actions_states = list(itertools.chain.from_iterable(self.joined_actions_states))
        # self.joined_actions_states = np.array(self.joined_actions_states)

        self.state_value = list(sequential_state_value.values())
        self.state_value = list(itertools.chain.from_iterable(self.state_value))
        # self.state_value = np.array(self.state_value)

        self.rewards = list(sequential_rewards.values())
        self.rewards = list(itertools.chain.from_iterable(self.rewards))
        # self.rewards = np.array(self.rewards)

        self.logprobs = list(sequential_logprobs.values())
        self.logprobs = list(itertools.chain.from_iterable(self.logprobs))
        # self.logprobs = np.array(self.logprobs)

        self.is_terminated = list(sequential_is_terminated.values())
        self.is_terminated = list(itertools.chain.from_iterable(self.is_terminated))
        # self.is_terminated = np.array(self.is_terminated)

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.joined_actions_states[:]
        del self.state_value[:]
        del self.rewards[:]
        del self.is_terminated[:]
        del self.logprobs[:]