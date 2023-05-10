import numpy
import numpy as np
from torch.distributions import Beta

from model.base_ac import BaseActorCritics
from model.network import QValueApproximation


class IndividualActorCritics(BaseActorCritics):
    def __init__(self, state_size=19, vocab_size=32):
        super().__init__(state_size=state_size, vocab_size=vocab_size)

    def forward(self):
        raise NotImplementedError

    def act(self, state, front_view):
        ba_sa, ba_acc = self.actor(state, front_view)

        sa_dist = Beta(ba_sa[0], ba_sa[1])
        acc_dist = Beta(ba_acc[0], ba_acc[1])

        sa_action = 2.0*sa_dist.sample() - 1.0
        acc_action = 2.0*acc_dist.sample() - 1.0

        sa_logprobs = sa_dist.log_prob((sa_action + 1.0)/2.0)
        acc_logprobs = acc_dist.log_prob((acc_action + 1.0)/2.0)
        logprobls = (sa_logprobs, acc_logprobs)

        return np.array([sa_action_tr.detach().numpy(), acc_action_tr.detach().numpy()]), logprobls

    def evaluate(self, state, front_view, action):
        ba_sa, ba_acc = self.actor(state, front_view)

        sa_dist = Beta(ba_sa[0], ba_sa[1])
        acc_dist = Beta(ba_acc[0], ba_acc[1])

        sa_logprobs = sa_dist.log_prob(action[0])
        sa_dist_entropy = sa_dist.entropy()

        acc_logprobs = acc_dist.log_prob(action[1])
        acc_dist_entropy = acc_dist.entropy()

        logprobls = (sa_logprobs, acc_logprobs)
        dist_entropy = (sa_dist_entropy, acc_dist_entropy)

        return logprobls, dist_entropy
