# Current Policy Configurations:
# ==========================================================================================
# Layer (type:depth-idx)                   Output Shape              Param #
# ==========================================================================================
# ├─Sequential: 1-1                        [-1, 128, 6, 6]           --
# |    └─Conv2d: 2-1                       [-1, 16, 100, 100]        736
# |    └─ReLU: 2-2                         [-1, 16, 100, 100]        --
# |    └─MaxPool2d: 2-3                    [-1, 16, 50, 50]          --
# |    └─Conv2d: 2-4                       [-1, 32, 50, 50]          4,640
# |    └─ReLU: 2-5                         [-1, 32, 50, 50]          --
# |    └─MaxPool2d: 2-6                    [-1, 32, 25, 25]          --
# |    └─Conv2d: 2-7                       [-1, 64, 25, 25]          18,496
# |    └─ReLU: 2-8                         [-1, 64, 25, 25]          --
# |    └─MaxPool2d: 2-9                    [-1, 64, 12, 12]          --
# |    └─Conv2d: 2-10                      [-1, 128, 12, 12]         73,856
# |    └─ReLU: 2-11                        [-1, 128, 12, 12]         --
# |    └─MaxPool2d: 2-12                   [-1, 128, 6, 6]           --
# ├─Linear: 1-2                            [-1]                      1,280
# ├─Linear: 1-3                            [-1]                      2,080
# ├─Embedding: 1-4                         [-1, 1, 64]               2,048
# ├─MultiheadAttention: 1-5                [-1, 4640, 64]            --
# ├─LSTM: 1-6                              [-1, 64, 8]               148,800
# ├─AvgPool1d: 1-7                         [-1, 1]                   --
# ├─Sequential: 1-8                        [-1]                      --
# |    └─Linear: 2-13                      [-1]                      2,080
# |    └─ReLU: 2-14                        [-1]                      --
# |    └─Linear: 2-15                      [-1]                      66
# |    └─Softplus: 2-16                    [-1]                      --
# ├─Sequential: 1-9                        [-1]                      --
# |    └─Linear: 2-17                      [-1]                      2,080
# |    └─ReLU: 2-18                        [-1]                      --
# |    └─Linear: 2-19                      [-1]                      66
# |    └─Softplus: 2-20                    [-1]                      --
# ==========================================================================================
# Total params: 256,228
# Trainable params: 256,228
# Non-trainable params: 0
# Total mult-adds (M): 41.13
# ==========================================================================================
# Input size (MB): 0.19
# Forward/backward pass size (MB): 2.28
# Params size (MB): 0.98
# Estimated Total Size (MB): 3.45
# ==========================================================================================
from algos.single.ppo_clip_mlp_beta_torch import SinglePPOClipMLPBetaAgent
from envs.multi_agents import MultiAgentsInterEnv, STATE_DIM
from envs.single_agent_intersection_lidar import SingleAgentInterLidarEnv
from torchsummary import summary
import torch

print("============================================================================================")
# set device to cpu or cuda
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")

if __name__ == '__main__':
    env = SingleAgentInterLidarEnv()
    obs = env.reset()
    state = torch.from_numpy(obs).to(device)

    K_epochs = 5  # update policy for K epochs in one PPO update
    batch_size = 512  # training batch size

    eps_clip = 0.2  # clip parameter for PPO
    gamma = 0.99  # discount factor

    lr_actor = 0.0003  # learning rate for actor network
    lr_critic = 0.001  # learning rate for critic network

    state_dim = env.observation_space.shape[0]

    action_dim = env.action_space.shape[0]

    ppo_agent = SinglePPOClipMLPBetaAgent(state_size=state_dim,
                                          batch_size=batch_size,
                                          lr_actor=lr_actor,
                                          lr_critic=lr_critic,
                                          gamma=gamma,
                                          k_epochs=K_epochs,
                                          eps_clip=eps_clip,
                                          hidden_size=256,
                                          action_size=action_dim,
                                          )
    print("==========================================================================================")
    print("Single Batch Summary: ")
    summary(ppo_agent.policy.actor, [state])

    # Testing with Batch Data
    # batch_images = torch.randn(50, 5, 100, 100)
    # batch_states = torch.randn(50, 19)
    # print("==========================================================================================")
    # print("100 Batch Summary: ")
    # summary(central_agent.actor, [batch_states, batch_images])
