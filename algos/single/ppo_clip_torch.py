# from collections import defaultdict
#
# import matplotlib.pyplot as plt
import torch
# from tensordict.nn import TensorDictModule
# from tensordict.nn.distributions import NormalParamExtractor
# from torch import nn
# from torchrl.collectors import SyncDataCollector
# from torchrl.data.replay_buffers import ReplayBuffer
# from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
# from torchrl.data.replay_buffers.storages import LazyTensorStorage
# from torchrl.envs import (
#     Compose,
#     DoubleToFloat,
#     ObservationNorm,
#     StepCounter,
#     TransformedEnv,
# )
from torchrl.envs.libs.gym import GymEnv
# from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type
# from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
# from torchrl.objectives import ClipPPOLoss
# from torchrl.objectives.value import GAE
# from tqdm import tqdm
from envs.single_agent_intersection import SingleAgentInterEnv, STATE_DIM

device = "cpu" if not torch.has_cuda else "cuda:0"
num_cells = 256  # number of cells in each layer
lr = 3e-4
max_grad_norm = 1.0

frame_skip = 1
frames_per_batch = 1000 // frame_skip
# For a complete training, bring the number of frames up to 1M
total_frames = 10_000 // frame_skip

sub_batch_size = 64  # cardinality of the sub-samples gathered from the current data in the inner loop
num_epochs = 10  # optimisation steps per batch of data collected
clip_epsilon = (
    0.2  # clip value for PPO loss: see the equation in the intro for more context.
)
gamma = 0.99
lmbda = 0.95
entropy_eps = 1e-4

env = SingleAgentInterEnv()

base_env = GymEnv(SingleAgentInterEnv, device=device, frame_skip=frame_skip)