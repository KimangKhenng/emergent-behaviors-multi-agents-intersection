import numpy as np
import torch
from torch.distributions import Beta
from torch import nn

from model.network import MLPPolicy, QValueApproximation

"""
MLPActor: Actor Critic with MLP
"""
# set device to cpu or cuda
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")


class MLPActorCritic(nn.Module):
    def __init__(self, state_size, hidden_size, num_layers, hidden_size_2, num_layers_2, output_size, action_size,
                 critics_hidden_size, critics_num_layers):
        super().__init__()
        self.actor = MLPPolicy(
            state_size=state_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            hidden_size_2=hidden_size_2,
            num_layers_2=num_layers_2,
            output_size=output_size,
        )
        self.critic = QValueApproximation(state_size + action_size, critics_hidden_size, critics_num_layers)

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        ba_sa, ba_acc = self.actor(state)

        sa_dist = Beta(ba_sa[0], ba_sa[1])
        acc_dist = Beta(ba_acc[0], ba_acc[1])

        sa_action = sa_dist.sample()
        acc_action = acc_dist.sample()

        sa_logprobs = sa_dist.log_prob(sa_action)
        acc_logprobs = acc_dist.log_prob(acc_action)
        logprobls = (sa_logprobs, acc_logprobs)

        sa_action = 2.0 * sa_action - 1.0
        acc_action = 2.0 * acc_action - 1.0

        acceleration = sa_action.detach().cpu().numpy()
        steering_angle = acc_action.detach().cpu().numpy()

        action = torch.Tensor([sa_action, acc_action]).to(device)
        joined_state = torch.cat((state, action), dim=0)

        q_value = self.critic(joined_state)

        return np.array([steering_angle, acceleration]), logprobls, q_value

    def evaluate(self, states, actions):
        ba_sa, ba_acc = self.actor(states)
        # actions = actions.reshape(int(len(actions)/2), 2)

        sa_dist = Beta(ba_sa[:, 0], ba_sa[:, 1])
        acc_dist = Beta(ba_acc[:, 0], ba_acc[:, 1])

        sa_logprobs = sa_dist.log_prob((actions[:, 0] + 1.0) / 2)
        sa_dist_entropy = sa_dist.entropy()
        # print("SA logprobs: ", sa_logprobs)

        acc_logprobs = acc_dist.log_prob((actions[:, 1] + 1.0) / 2)
        acc_dist_entropy = acc_dist.entropy()

        joined_state = torch.cat((states, actions), dim=1).to(device)
        q_value = self.critic(joined_state)

        logprobs = torch.cat([sa_logprobs.unsqueeze(1), acc_logprobs.unsqueeze(1)], dim=1)
        dist_entropy = torch.cat([sa_dist_entropy.unsqueeze(1), acc_dist_entropy.unsqueeze(1)], dim=1)

        return logprobs, dist_entropy, q_value
