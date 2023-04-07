import torch
import torch.nn as nn
import torch.optim as optim

from model.network import Policy


class PPOAgents(nn.Module):
    def __init__(self, agent_id, state_size=19, image_height=84, image_width=84, action_size=2):
        # Initialize the network and optimizer
        super().__init__()
        self.agent_id = agent_id
        self.policy = Policy(state_size=state_size, image_width=image_width, image_height=image_height,
                             action_size=action_size)

    def train_agent(self, obs, episode):
        actions = self.policy(torch.from_numpy(obs['image']), torch.from_numpy(obs['state']))
        return actions
