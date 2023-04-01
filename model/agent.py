import torch
import torch.nn as nn
import torch.optim as optim

from model.network import Policy


class PPOAgent:
    def __init__(self, state_size, image_height, image_width, action_size):
        # Initialize the network and optimizer
        self.model = Policy(state_size, image_height, image_width, action_size)
