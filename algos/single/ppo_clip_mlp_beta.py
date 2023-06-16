import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Beta

################################## set device ##################################
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


################################## PPO Policy ##################################
class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]


class BetaPolicy(nn.Module):
    def __init__(self, input_size, hidden_size, action_size):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.Tanh(),
            # nn.BatchNorm1d(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            # nn.BatchNorm1d(hidden_size),
        )

        self.alpha = nn.Sequential(
            nn.Linear(hidden_size, action_size, bias=True),
            nn.Softplus()
        )
        # self.alpha.weight.data.mul_(0.125)

        self.beta = nn.Sequential(
            nn.Linear(hidden_size, action_size, bias=True),
            nn.Softplus()
        )
        # self.beta.weight.data.mul_(0.125)

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
        ab_sa = self.alpha(self.actor(state)) + 1.0
        ab_acc = self.beta(self.actor(state)) + 1.0

        sa_dist = Beta(ab_sa[0], ab_sa[1])
        acc_dist = Beta(ab_acc[0], ab_acc[1])

        sa_action = 2.0 * sa_dist.sample() - 1.0
        acc_action = 2.0 * acc_dist.sample() - 1.0

        sa_logprobs = sa_dist.log_prob((sa_action + 1.0) / 2.0)
        acc_logprobs = acc_dist.log_prob((acc_action + 1.0) / 2.0)
        logprobls = (sa_logprobs, acc_logprobs)

        state_val = self.critic(state)
        return np.array([sa_action.detach().cpu().numpy(), acc_action.detach().cpu().numpy()]), logprobls, state_val

    def evaluate(self, state, actions):
        ab_sa = self.alpha(self.actor(state)) + 1.0
        ab_acc = self.beta(self.actor(state)) + 1.0

        sa_dist = Beta(ab_sa[:, 0], ab_sa[:, 1])
        acc_dist = Beta(ab_acc[:, 0], ab_acc[:, 1])

        sa_logprobs = sa_dist.log_prob((actions[:, 0] + 1.0) / 2)
        sa_dist_entropy = sa_dist.entropy()
        # print("SA logprobs: ", sa_logprobs)

        acc_logprobs = acc_dist.log_prob((actions[:, 1] + 1.0) / 2)
        acc_dist_entropy = acc_dist.entropy()

        action_state_value = self.critic(state)

        logprobs = torch.cat([sa_logprobs.unsqueeze(1), acc_logprobs.unsqueeze(1)], dim=1)
        dist_entropy = torch.cat([sa_dist_entropy.unsqueeze(1), acc_dist_entropy.unsqueeze(1)], dim=1)

        return logprobs, dist_entropy, action_state_value


class SPPOClipMLPBeta:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, hidden_dim):

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.buffer = RolloutBuffer()

        self.policy = BetaPolicy(input_size=state_dim, action_size=action_dim, hidden_size=hidden_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = BetaPolicy(input_size=state_dim, action_size=action_dim, hidden_size=hidden_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            actions, action_logprob, state_val = self.policy_old.act(state)

        self.buffer.states.append(state)
        self.buffer.actions.append(torch.Tensor(actions))
        self.buffer.logprobs.append(torch.Tensor(action_logprob))
        self.buffer.state_values.append(state_val)

        return actions

    def update(self):
        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)
        old_state_values = torch.squeeze(torch.stack(self.buffer.state_values, dim=0)).detach().to(device)

        # calculate advantages
        advantages = rewards.detach() - old_state_values.detach()

        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, dist_entropy, state_values = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            # print(logprobs.shape, old_logprobs.shape)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            # print(ratios.shape, advantages.shape)
            advantages = advantages.unsqueeze(1)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            # print(surr1.shape, surr2.shape, state_values.shape, rewards.shape, dist_entropy.shape)
            # print(surr1.shape, surr2.shape, state_values.shape, rewards.shape, dist_entropy.shape)
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            advantages = advantages.squeeze(1)

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
