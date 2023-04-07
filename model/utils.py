import random

import torch

from model.agent import get_available_agents, get_agent_info

if __name__ == '__main__':
    # Define the number of time steps and agents
    num_time_steps = 5
    num_agents = 5

    # Generate random rewards and done conditions for each agent at each time step
    rewards = [{f'agent{i}': random.randint(0, 10) for i in range(num_agents)} for t in range(num_time_steps)]
    dones = [{f'agent{i}': random.choice([True, False]) for i in range(num_agents)} for t in range(num_time_steps)]

    print(f"rewards: {rewards}")
    print(f"dones: {dones}")

    # Extract the rewards and done conditions for a specific agent
    available_agents = get_available_agents(rewards, dones)
    print(f"available_agents: {available_agents}")
