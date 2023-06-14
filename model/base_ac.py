from torch import nn

from model.network import Policy, LSTMPolicy, MLPPolicy


class BaseActorCritics(nn.Module):
    def __init__(self, state_size=19):
        super().__init__()
        # self.actor = MLPPolicy(state_size=state_size, hidden_size=128, hidden_size_2=64, hidden_layer=2)
        self.actor = Policy(state_size=state_size)
