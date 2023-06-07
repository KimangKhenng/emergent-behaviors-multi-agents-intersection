import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Fu
from torch.distributions import Beta

from envs.multi_agents import STATE_DIM


# Define hybrid policy architecture
class Policy(nn.Module):

    def __init__(self, state_size=STATE_DIM, vocab_size=32):
        super(Policy, self).__init__()
        # Process state information
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
        self.lstm = nn.LSTM(4640, hidden_size=8, num_layers=1, batch_first=True)

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


class LSTMPolicy(nn.Module):
    def __init__(self, state_size=STATE_DIM):
        super(LSTMPolicy, self).__init__()
        # Process state information
        self.fc1 = nn.Linear(in_features=state_size, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)

        # Process front view image
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1),
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
            # nn.Flatten(start_dim=0)
        )

        # 1D Average pooling layer
        self.avg_pool = nn.AvgPool1d(kernel_size=6, stride=4)

        # Linear layer for attention
        self.linear_attn = nn.Linear(3200 + 32, 64)

        # LSTM layer
        self.lstm = nn.LSTM(64, hidden_size=8, num_layers=1, batch_first=True)

        # Final dense layer for steering angle
        self.final_dense_sa = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2),
        )

        # Final dense layer for acceleration
        self.final_dense_acc = nn.Sequential(
            nn.Linear(in_features=64, out_features=32),
            nn.ReLU(),
            nn.Linear(in_features=32, out_features=2),
        )

    def forward(self, x_vec, x_img):
        # Image processing
        x_img = self.conv(x_img)
        # print(x_img.ndim)
        if x_img.ndim == 3:
            # print("Single Input")
            x_img = torch.flatten(x_img)
        if x_img.ndim == 4:
            # print("Batch Input")
            x_img = torch.flatten(x_img, start_dim=1)
        # Processing state
        x_vec = Fu.relu(self.fc1(x_vec))
        x_vec = Fu.relu(self.fc2(x_vec))

        x = 0
        if x_img.ndim == 1:
            x = torch.cat((x_img, x_vec))
        if x_img.ndim == 2:
            x = torch.cat((x_img, x_vec), dim=1)

        x = self.linear_attn(x)

        # Output alpha and beta value for steering angle and acceleration to be used in Beta distribution
        ba_sa = torch.exp(self.final_dense_sa(x))
        ba_sa = 1 + ba_sa
        ba_acc = torch.exp(self.final_dense_acc(x))
        ba_acc = 1 + ba_acc
        return ba_sa, ba_acc


class QValueApproximation(nn.Module):
    def __init__(self, state_size, joined_actions_size):
        super().__init__()
        self.fc1 = nn.Linear(in_features=state_size + joined_actions_size, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)
        self.fc3 = nn.Linear(in_features=32, out_features=1)

    def forward(self, joined_state_action):
        x = Fu.relu(self.fc1(joined_state_action))
        x = Fu.relu(self.fc2(x))
        x = Fu.tanh(self.fc3(x))
        return x
