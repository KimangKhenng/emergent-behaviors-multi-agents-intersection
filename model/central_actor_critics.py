import numpy as np
import torch
from torch.distributions import Beta

from model.base_ac import BaseActorCritics
from model.network import QValueApproximation


class CentralActorCritics(BaseActorCritics):
    def __init__(self, state_size=19, num_agents=8):
        super().__init__(state_size=state_size)
        self.critic = QValueApproximation(state_size=state_size * num_agents, joined_actions_size=2 * num_agents)

    def forward(self):
        raise NotImplementedError

    def act(self, state, front_view, joined_state_action):
        ba_sa, ba_acc = self.actor(state, front_view)

        sa_dist = Beta(ba_sa[0], ba_sa[1])
        acc_dist = Beta(ba_acc[0], ba_acc[1])

        sa_action = 2.0*sa_dist.sample() - 1.0
        acc_action = 2.0*acc_dist.sample() - 1.0

        sa_logprobs = sa_dist.log_prob((sa_action + 1.0)/2.0)
        acc_logprobs = acc_dist.log_prob((acc_action + 1.0)/2.0)

        action_state_value = self.critic(joined_state_action)

        return np.array([sa_action.detach().numpy(), acc_action.detach().numpy()]), (
            sa_logprobs, acc_logprobs), action_state_value

    def evaluate(self, state, front_view, actions, joined_state_action):
        ba_sa, ba_acc = self.actor(state, front_view)
        actions = actions.reshape(int(len(actions)/2), 2)

        sa_dist = Beta(ba_sa[:, 0], ba_sa[:, 1])
        acc_dist = Beta(ba_acc[:, 0], ba_acc[:, 1])

        sa_logprobs = sa_dist.log_prob((actions[:, 0] + 1.0)/2)
        sa_dist_entropy = sa_dist.entropy()
        # print("SA logprobs: ", sa_logprobs)

        acc_logprobs = acc_dist.log_prob((actions[:, 1] + 1.0)/2)
        acc_dist_entropy = acc_dist.entropy()

        action_state_value = self.critic(joined_state_action)

        logprobs = torch.cat([sa_logprobs.unsqueeze(1), acc_logprobs.unsqueeze(1)], dim=1)
        # print("Logprobs: ", logprobs)
        # print("Logprobs shape: ", logprobs.shape)
        # dist_entropy = (sa_dist_entropy, acc_dist_entropy)
        dist_entropy = torch.cat([sa_dist_entropy.unsqueeze(1), acc_dist_entropy.unsqueeze(1)], dim=1)

        return logprobs, dist_entropy, action_state_value
