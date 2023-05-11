from torch import nn

from model.network import Policy, LSTMPolicy


class BaseActorCritics(nn.Module):
    def __init__(self, state_size=19, vocab_size=32):
        super().__init__()
        self.actor = Policy(state_size)

