from metadrive import SafeMetaDriveEnv
import torch
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np
import torch.optim as optim

from model.network import Policy

NUM_AGENTS = 4
NUM_EPISODE = 1000

env = SafeMetaDriveEnv(config={
    "map_config": {
        "lane_num": 3
    },
    "vehicle_config": {
        "show_lidar": True,
        "image_source": "rgb_camera",
        "rgb_camera": (84, 84)
    },
    "image_observation": True
})
obs = env.reset()


# print("Sate Length: ", obs['state'].shape)
# print("Image Shape: ", obs['image'].shape)

# Define the policy gradient algorithm
def policy_gradient(num_episodes, gamma, lr, hidden_dim):
    # Create the policy network
    policy_net = Policy(state_size=19, image_width=84, image_height=84, action_size=2)

    # Create the optimizer
    optimizer = optim.Adam(policy_net.parameters(), lr=lr)

    # Run the training loop
    for episode in range(num_episodes):

        # Initialize the episode
        state = env.reset()
        episode_rewards = []
        episode_log_probs = []

        # Collect experience for the episode
        while True:
            # Get the action from the policy
            action_probs = policy_net(torch.from_numpy(state['image']).permute(3, 2, 0, 1),
                                      torch.from_numpy(state['state']))
            dist = torch.distributions.Categorical(action_probs)
            action = dist.sample()

            # Take the action in the environment
            print("Actions: ", action)
            next_state, reward, done, _ = env.step(action.detach().numpy())

            # Calculate the log probability of the action
            log_prob = dist.log_prob(action)

            # Store the reward and log probability
            episode_rewards.append(reward)
            episode_log_probs.append(log_prob)

            # Update the state
            state = next_state

            # Check if the episode is over
            if done:
                break

        # Compute the discounted rewards
        discounted_rewards = []
        R = 0
        for reward in episode_rewards[::-1]:
            R = reward + gamma * R
            discounted_rewards.insert(0, R)

        # Normalize the rewards
        discounted_rewards = torch.FloatTensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-6)

        # Compute the loss
        policy_loss = []
        for log_prob, reward in zip(episode_log_probs, discounted_rewards):
            policy_loss.append(-log_prob * reward)
        policy_loss = torch.stack(policy_loss).sum()

        # Optimize the policy
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # Print the episode rewards
        print(f"Episode {episode + 1}: Total reward = {sum(episode_rewards)}")

    env.close()


if __name__ == '__main__':
    num_episodes = 100
    gamma = 0.99
    lr = 0.001
    hidden_dim = 64
    policy_gradient(num_episodes, gamma, lr, hidden_dim)
