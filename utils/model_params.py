# Current Policy Configurations:
# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# ├─Sequential: 1-1                        [-1, 5, 12]               --
# |    └─Conv2d: 2-1                       [-1, 88, 200]             448
# |    └─ReLU: 2-2                         [-1, 88, 200]             --
# |    └─MaxPool2d: 2-3                    [-1, 44, 100]             --
# |    └─Conv2d: 2-4                       [-1, 44, 100]             4,640
# |    └─ReLU: 2-5                         [-1, 44, 100]             --
# |    └─MaxPool2d: 2-6                    [-1, 22, 50]              --
# |    └─Conv2d: 2-7                       [-1, 22, 50]              18,496
# |    └─ReLU: 2-8                         [-1, 22, 50]              --
# |    └─MaxPool2d: 2-9                    [-1, 11, 25]              --
# |    └─Conv2d: 2-10                      [-1, 11, 25]              73,856
# |    └─ReLU: 2-11                        [-1, 11, 25]              --
# |    └─MaxPool2d: 2-12                   [-1, 5, 12]               --
# ├─Linear: 1-2                            [-1]                      1,280
# ├─Linear: 1-3                            [-1]                      2,080
# ├─Embedding: 1-4                         [-1, 1, 256]              8,192
# ├─MultiheadAttention: 1-5                [-1, 7712, 256]           --
# ├─LSTM: 1-6                              [-1, 256, 32]             991,488
# ├─AvgPool1d: 1-7                         [-1, 7]                   --
# ├─Sequential: 1-8                        [-1]                      --
# |    └─Linear: 2-13                      [-1]                      358,600
# |    └─ReLU: 2-14                        [-1]                      --
# |    └─Linear: 2-15                      [-1]                      20,100
# |    └─ReLU: 2-16                        [-1]                      --
# |    └─Linear: 2-17                      [-1]                      202
# ├─Sequential: 1-9                        [-1]                      --
# |    └─Linear: 2-18                      [-1]                      358,600
# |    └─ReLU: 2-19                        [-1]                      --
# |    └─Linear: 2-20                      [-1]                      20,100
# |    └─ReLU: 2-21                        [-1]                      --
# |    └─Linear: 2-22                      [-1]                      202
# ==========================================================================================
# Total params: 1,858,284
# Trainable params: 1,858,284
# Non-trainable params: 0
# Total mult-adds (M): 6.19
# ==========================================================================================
# Input size (MB): 0.20
# Forward/backward pass size (MB): 0.24
# Params size (MB): 7.09
# Estimated Total Size (MB): 7.53
# ==========================================================================================
from envs.multi_agents import MultiAgentsEnv
from torchsummary import summary
import torch
from model.indi_actor_critics import IndividualActorCritics

if __name__ == '__main__':
    env = MultiAgentsEnv(num_agents=8)
    obs = env.reset()
    state = torch.from_numpy(obs['agent3']['state'])
    sample_image = obs['agent0']['image']
    sample_image = sample_image[:, :, :, -1]
    central_agent = IndividualActorCritics(state_size=19)
    summary(central_agent.actor, [state, torch.from_numpy(sample_image).permute(2, 0, 1)])