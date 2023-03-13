import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.data import BatchSampler, SubsetRandomSampler
from model.network import ActorCritic


class PPOAgent:
    def __init__(self, agent_id, num_inputs, num_actions, hidden_size, rnn_layers=1,
                 clip_param=0.2, ppo_epoch=10, num_mini_batch=32, value_loss_coef=0.5,
                 entropy_coef=0.01, lr=0.0007, eps=1e-5, max_grad_norm=0.5):
        # Initialize the network and optimizer
        self.model = ActorCritic(num_inputs, num_actions, hidden_size, rnn_layers)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, eps=eps)

        # Set the hyperparameters for PPO
        self.clip_param = clip_param
        self.ppo_epoch = ppo_epoch
        self.num_mini_batch = num_mini_batch
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.agent_id = agent_id

    def select_action(self, state, rnn_state, comm_in=None):
        # Process the state and communication inputs
        state = torch.FloatTensor(state).unsqueeze(0)
        if comm_in is not None:
            comm_in = torch.FloatTensor(comm_in).unsqueeze(0)

        # Compute the actor output, critic output, and LSTM hidden state
        dist, value, rnn_state = self.model((state[:, 3:], state[:, :3]), rnn_state, comm_in)
        actor_out = dist.probs

        # Sample an action from the output distribution
        dist = Categorical(actor_out)
        action = dist.sample()

        return action.item(), rnn_state

    def update(self, rollouts, comm_matrix):
        # Compute the advantages and returns
        advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
        returns = rollouts.returns[:-1]

        # Compute the mini-batch size
        mini_batch_size = rollouts.num_steps // self.num_mini_batch

        # Iterate over the mini-batches
        for _ in range(self.ppo_epoch):
            # Shuffle the mini-batches
            sampler = BatchSampler(SubsetRandomSampler(range(rollouts.num_steps - 1)), mini_batch_size)
            for indices in sampler:
                # Compute the loss for the current mini-batch
                states = rollouts.states[:-1][indices]
                actions = rollouts.actions[indices]
                old_action_probs = rollouts.action_probs[indices].detach()
                old_values = rollouts.value_preds[:-1][indices].detach()

                # Compute the communication inputs and outputs for the current mini-batch
                comm_in = comm_matrix[indices, self.agent_id, :]
                comm_out = torch.zeros(comm_in.size())
                for i in range(len(indices)):
                    comm_out[i] = self.model.comm_linear(self.model.rnn(torch.cat(
                        [self.model.actor_linear.weight.data[:, actions[i]], comm_in[i].unsqueeze(0)], dim=1).unsqueeze(
                        0),
                        rollouts.rnn_states[indices[i]][0])[0][0][0])

                # Compute the actor and critic outputs for the current mini-batch
                dist, value, _ = self.model((states[:, 3:], states[:, :3]), rollouts.rnn_states[indices], comm_out)
                action_probs = dist.log_prob(actions)
                entropy = dist.entropy().mean()

                ratio = torch.exp(action_probs - old_action_probs)
                surr1 = ratio * advantages[indices]
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages[indices]
                actor_loss = -torch.min(surr1, surr2).mean()
                critic_loss = F.smooth_l1_loss(value, returns[indices])
                loss = actor_loss + self.value_loss_coef * critic_loss - self.entropy_coef * entropy

                # Compute the gradients and update the model parameters
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

    def save(self, path):
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        self.model.load_state_dict(torch.load(path))
