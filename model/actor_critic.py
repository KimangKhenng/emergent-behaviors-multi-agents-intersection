import numpy as np
import torch
from torch.distributions import Beta

from model.base_ac import BaseActorCritics
from model.network import QValueApproximation


class ActorCritics(BaseActorCritics):
    def __init__(self, state_size=21):
        super().__init__(state_size=state_size)
        self.critic = QValueApproximation(state_size=state_size, joined_actions_size=0)

    def forward(self):
        raise NotImplementedError

    def act(self, state, front_view):
        ba_sa, ba_acc = self.actor(state, front_view)
        # print("Beta Alpha Steering Angle: ", ba_sa)
        # print("Beta Alpha Acceleration: ", ba_acc)

        sa_dist = Beta(ba_sa[0], ba_sa[1])
        acc_dist = Beta(ba_acc[0], ba_acc[1])

        sa_action = 2.0 * sa_dist.sample() - 1.0
        acc_action = 2.0 * acc_dist.sample() - 1.0

        sa_logprobs = sa_dist.log_prob((sa_action + 1.0) / 2.0)
        acc_logprobs = acc_dist.log_prob((acc_action + 1.0) / 2.0)

        # print("State: ", state)
        # print("Acceleration: ", acc_action)
        # print("Steering Angle: ", sa_action)
        # joined_state_action = torch.add(state, acc_action)
        # joined_state_action = torch.add(joined_state_action, sa_action)
        # print("Joined State Action: ", joined_state_action)

        action_state_value = self.critic(state)

        acceleration = sa_action.detach().cpu().numpy()
        steering_angle = acc_action.detach().cpu().numpy()

        return np.array([acceleration, steering_angle]), (
            sa_logprobs, acc_logprobs), action_state_value

    def evaluate(self, state, front_view, actions):
        ba_sa, ba_acc = self.actor(state, front_view)
        # actions = actions.reshape(int(len(actions)/2), 2)

        sa_dist = Beta(ba_sa[:, 0], ba_sa[:, 1])
        acc_dist = Beta(ba_acc[:, 0], ba_acc[:, 1])

        sa_logprobs = sa_dist.log_prob((actions[:, 0] + 1.0) / 2)
        sa_dist_entropy = sa_dist.entropy()
        # print("SA logprobs: ", sa_logprobs)

        acc_logprobs = acc_dist.log_prob((actions[:, 1] + 1.0) / 2)
        acc_dist_entropy = acc_dist.entropy()

        action_state_value = self.critic(state)

        logprobs = torch.cat([sa_logprobs.unsqueeze(1), acc_logprobs.unsqueeze(1)], dim=1)
        dist_entropy = torch.cat([sa_dist_entropy.unsqueeze(1), acc_dist_entropy.unsqueeze(1)], dim=1)

        return logprobs, dist_entropy, action_state_value
