import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fu
from torch.distributions import Beta

from envs.multi_agents import STATE_DIM


# Define hybrid policy architecture
class CNNPolicy(nn.Module):

    def __init__(self, state_size=STATE_DIM, vocab_size=32):
        # Process state information
        super().__init__()
        self.fc1 = nn.Linear(in_features=state_size, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)

        # Process front view image
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            # nn.ReLU(),
            # nn.MaxPool2d(kernel_size=2, stride=2),
            # nn.Flatten(start_dim=0)
        )

        # self.conv = self.conv.cuda()

        # 1D Average pooling layer
        self.avg_pool = nn.AvgPool1d(kernel_size=6, stride=4)

        # # Batch normalization layers
        # self.bn1 = nn.BatchNorm1d(23040)

        #  Embedding layer
        self.embedding = nn.Embedding(vocab_size, 64)

        # # Multi-head attention layer
        self.self_attn = nn.MultiheadAttention(64, num_heads=8, dropout=0.1)
        # Linear layer for attention
        # self.linear_attn = nn.Linear(64, 64)

        # LSTM layer
        self.lstm = nn.LSTM(3232, hidden_size=8, num_layers=1, batch_first=True)

        # Final dense layer for steering angle
        self.final_dense_sa = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2),
            nn.Softplus()
            # nn.ReLU(),
            # nn.Linear(in_features=100, out_features=2),
        )

        # Final dense layer for acceleration
        self.final_dense_acc = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2),
            nn.Softplus()
            # nn.ReLU(),
            # nn.Linear(in_features=100, out_features=2),
        )

    def forward(self, x_vec, x_img):
        # Image processing
        x_img = self.conv(x_img)
        # print(x_img.ndim)
        # print(x_img.shape)
        # print(x_img.shape[0])
        if x_img.shape[0] == 1:
            # print("Single Input")
            x_img = torch.flatten(x_img)
        else:
            # print("Batch Input")
            x_img = torch.flatten(x_img, start_dim=1)
        # Processing state
        x_vec = Fu.relu(self.fc1(x_vec))
        x_vec = Fu.relu(self.fc2(x_vec))

        # print("X Vec Shape: ", x_vec.shape)
        # print("X Img Shape: ", x_img.shape)
        # Concatenate image output and vector output
        x = 0
        if x_img.ndim == 1:
            x = torch.cat((x_img, x_vec))
        if x_img.ndim == 2:
            x = torch.cat((x_img, x_vec), dim=1)

        x = x.long()
        x = x.unsqueeze(-1)
        embedded = self.embedding(x)
        # embedded = embedded.to_dense()
        # print("Embedded Shape: ", embedded.shape)
        if embedded.ndim == 3:
            x = embedded.permute(1, 0, 2)
        if embedded.ndim == 4:
            x = embedded.permute(2, 0, 1, 3)
            x = x.squeeze(0)

        # print("After Shape: ", x.shape)

        # Apply self-attention with residual connection and layer normalization
        attn_output, _ = self.self_attn(x, x, x)
        attn_output = Fu.layer_norm(x + attn_output, [attn_output.shape[-1]])

        # Reshape the self-attention output to match the LSTM input shape
        # Shape: (batch size, sequence length, embedding dimension)
        attn_output = attn_output.permute(0, 2, 1)

        # # LSTM layer
        # x = x.unsqueeze(0)  # Add sequence length dimension
        lstm_output, _ = self.lstm(attn_output)
        lstm_output = lstm_output.squeeze(0)  # Remove sequence length dimension

        # Flatten lstm output
        lstm_output = self.avg_pool(lstm_output)
        # print("LSTM Output Shape: ", lstm_output.shape)
        if lstm_output.ndim == 2:
            lstm_output = torch.flatten(lstm_output)
        else:
            lstm_output = torch.flatten(lstm_output, start_dim=1)

        # Output alpha and beta value for steering angle and acceleration to be used in Beta distribution
        ba_sa = self.final_dense_sa(lstm_output)
        ba_sa = 1 + ba_sa
        ba_acc = self.final_dense_acc(lstm_output)
        ba_acc = 1 + ba_acc
        return ba_sa, ba_acc


class MLPPolicy(nn.Module):
    """
    Multi-layer perceptron policy network
    state_size: size of the state vector
    hidden_size: size of the first hidden layer
    hidden_size_2: size of the second hidden layer
    num_layers: number of layers in the first hidden layer
    num_layers_2: number of layers in the second hidden layer
    output_size: size of the output vector
    """

    def __init__(
            self,
            state_size=259,
            hidden_size=64,
            num_layers=2,
            hidden_size_2=64,
            num_layers_2=2,
            output_size=2,
    ):
        super().__init__()
        layers = [nn.Linear(in_features=state_size, out_features=hidden_size), nn.ReLU()]

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
            layers.append(nn.ReLU())

        self.fc1 = nn.Sequential(*layers)

        self.embedding = nn.Embedding(32, hidden_size_2)

        layers_2 = [nn.Linear(in_features=hidden_size_2, out_features=hidden_size_2), nn.ReLU()]
        for _ in range(num_layers_2 - 1):
            layers_2.append(nn.Linear(in_features=hidden_size_2, out_features=hidden_size_2))
            layers_2.append(nn.ReLU())
        layers_2.append(nn.Linear(in_features=hidden_size_2, out_features=output_size))
        layers_2.append(nn.Softplus())
        self.fc_sa = nn.Sequential(*layers_2)

        layers_3 = [nn.Linear(in_features=hidden_size_2, out_features=hidden_size_2), nn.ReLU()]
        for _ in range(num_layers_2 - 1):
            layers_3.append(nn.Linear(in_features=hidden_size_2, out_features=hidden_size_2))
            layers_3.append(nn.ReLU())
        layers_3.append(nn.Linear(in_features=hidden_size_2, out_features=output_size))
        layers_3.append(nn.Softplus())
        self.fc_acc = nn.Sequential(*layers_3)

        self.self_attn = nn.MultiheadAttention(hidden_size_2, num_heads=8, dropout=0.1)
        self.lstm = nn.LSTM(hidden_size_2 * 2, hidden_size=8, num_layers=2, batch_first=True)
        self.avg_pool = nn.AvgPool1d(kernel_size=6, stride=4)

    def forward(self, x):
        # print("Before Shape: ", x.shape)
        x = self.fc1(x)
        # print("After Shape: ", x.shape)
        x = x.long()
        x = x.unsqueeze(-1)
        embedded = self.embedding(x)
        # print("Embedded Shape: ", embedded.shape)
        if embedded.ndim == 3:
            x = embedded.permute(1, 0, 2)
        if embedded.ndim == 4:
            x = embedded.permute(2, 0, 1, 3)
            x = x.squeeze(0)
        # print("After Shape: ", x.shape)

        # print("After Shape: ", x.shape)

        # Apply self-attention with residual connection and layer normalization
        attn_output, _ = self.self_attn(x, x, x)
        attn_output = Fu.layer_norm(x + attn_output, [attn_output.shape[-1]])
        # print("Attention Shape: ", attn_output.shape)

        # Reshape the self-attention output to match the LSTM input shape
        # Shape: (batch size, sequence length, embedding dimension)
        attn_output = attn_output.permute(0, 2, 1)
        # print("After Shape: ", attn_output.shape)

        # # LSTM layer
        # x = x.unsqueeze(0)  # Add sequence length dimension
        lstm_output, _ = self.lstm(attn_output)
        lstm_output = lstm_output.squeeze(0)  # Remove sequence length dimension

        # Flatten lstm output
        lstm_output = self.avg_pool(lstm_output)
        # print("LSTM Output Shape: ", lstm_output.shape)
        if lstm_output.ndim == 2:
            lstm_output = torch.flatten(lstm_output)
        else:
            lstm_output = torch.flatten(lstm_output, start_dim=1)
        # print("LSTM Output Shape: ", lstm_output.shape)

        ba_sa = self.fc_sa(lstm_output) + 1.0
        ba_acc = self.fc_acc(lstm_output) + 1.0
        return ba_sa, ba_acc


class BetaPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.LeakyReLU(),
            # nn.BatchNorm1d(hidden_size),
        )

        self.alpha = nn.Sequential(
            nn.Linear(hidden_size, action_size, bias=False),
            nn.Softplus(),
        )

        self.beta = nn.Sequential(
            nn.Linear(hidden_size, action_size, bias=False),
            nn.Softplus(),
        )

        self.critic = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        alpha = self.alpha(self.actor(state)) + 1.0
        beta = self.beta(self.actor(state)) + 1.0
        # policy distribution - using Beta here which is a lot more efficient than Gaussian
        policy_dist = torch.distributions.Beta(alpha, beta)
        action = policy_dist.sample()
        log_prob = policy_dist.log_prob(action).sum(-1).unsqueeze(-1)
        state_val = self.critic(state)
        return action.detach(), log_prob.detach(), state_val.detach()

    def evaluate(self, state, action):
        alpha = self.alpha(self.actor(state)) + 1.0
        beta = self.beta(self.actor(state)) + 1.0
        # policy distribution - using Beta here which is a lot more efficient than Gaussian

        dist = torch.distributions.Beta(alpha, beta)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy


class QValueApproximation(nn.Module):
    """
    Q-value approximation network
    input_size: size of input
    hidden_size: size of hidden layer
    num_layers: number of hidden layers
    output_size: size of output
    """

    def __init__(self, input_size, hidden_size, num_layers):
        super().__init__()
        layers = [nn.Linear(in_features=input_size, out_features=hidden_size), nn.ReLU()]
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(in_features=hidden_size, out_features=hidden_size))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(in_features=hidden_size, out_features=1))

        self.fc = nn.Sequential(*layers)

    def forward(self, joined_state_action):
        x = self.fc(joined_state_action)
        return x
