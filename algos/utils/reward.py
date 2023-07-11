# def monte_carlo_state_rewards(rewards, termination_states, gamma):
#     state_rewards = []
#     discounted_reward = 0
#
#     for reward, is_terminated in zip(reversed(rewards), reversed(termination_states)):
#         discounted_reward = reward + gamma * discounted_reward
#         if not is_terminated:
#             state_rewards.insert(0, discounted_reward)
#         else:
#             state_rewards.insert(0, reward)
#             discounted_reward = 0
#
#     return state_rewards
import numpy as np


def meanfield_approximation(values):
    n = len(values)
    mean = np.mean(values)
    variance = np.var(values)
    meanfield_value = mean + variance / (2 * n)
    return meanfield_value

def monte_carlo_state_rewards(rewards, termination_states, gamma):
    state_rewards = []
    discounted_reward = 0
    for reward, is_terminated in zip(reversed(rewards),
                                     reversed(termination_states)):
        if is_terminated:
            discounted_reward = 0
        discounted_reward = reward + (gamma * discounted_reward)
        state_rewards.insert(0, discounted_reward)
    return state_rewards


def generalized_advantage_estimation(rewards, termination_states, gamma, lambda_=0.95):
    gae_values = []
    next_gae = 0

    for reward, is_terminated in zip(reversed(rewards), reversed(termination_states)):
        delta = reward + gamma * next_gae - gae_values[0] if gae_values else reward
        next_gae = delta + gamma * lambda_ * next_gae * (not is_terminated)
        gae_values.insert(0, delta + gae_values[0] if gae_values else delta)

    return gae_values
