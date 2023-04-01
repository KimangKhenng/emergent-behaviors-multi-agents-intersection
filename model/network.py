import torch
import torch.nn as nn
import torch.nn.functional as Fu


# Define hybrid policy architecture
class Policy(nn.Module):
    FLATTENED_FEATURES = None

    def __init__(self, state_size, image_height, image_width, action_size):
        super(Policy, self).__init__()
        # Process state information
        self.fc1 = nn.Linear(in_features=state_size, out_features=64)
        self.fc2 = nn.Linear(in_features=64, out_features=32)

        # Process front view image
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # LSTM layer
        self.lstm = nn.LSTM(42368, hidden_size=16, num_layers=1, batch_first=True)

        # Final dense layer for classification
        self.final_dense = nn.Linear(in_features=16, out_features=action_size)
        self.softmax = nn.Softmax(dim=-1)

        # Compute the number of features in the flattened output
        with torch.no_grad():
            dummy_input_img = torch.randn(1, 3, image_height, image_width)
            dummy_output_img = self.conv2(self.pool(Fu.relu(self.conv1(dummy_input_img))))
            Policy.FLATTENED_FEATURES = dummy_output_img.numel()

    def forward(self, x_img, x_vec):
        # Image processing
        x_img = Fu.relu(self.conv1(x_img))
        x_img = self.pool(x_img)
        x_img = Fu.relu(self.conv2(x_img))
        x_img = self.pool(x_img)
        x_img = torch.flatten(x_img)

        # Processing another input vector
        x_vec = Fu.relu(self.fc1(x_vec))
        x_vec = Fu.relu(self.fc2(x_vec))

        # Concatenate image output and vector output
        x = torch.cat((x_img, x_vec))

        # LSTM layer
        x = x.unsqueeze(0)  # Add sequence length dimension
        x, _ = self.lstm(x)
        x = x.squeeze(0)  # Remove sequence length dimension

        # Final dense layer for classification
        x = self.softmax(self.final_dense(x))
        return x
