import os
import numpy as np
import itertools
from collections import deque, namedtuple
import sys
import copy
import time
import random
import math
from or_gym.envs.finance.discrete_portfolio_opt import DiscretePortfolioOptEnv
import json
import matplotlib
matplotlib.use('tkagg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


# ===== Create Output Folders =====
os.makedirs("models", exist_ok=True)
os.makedirs("plots", exist_ok=True)
os.makedirs("evaluation", exist_ok=True)

# Basic settings
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)

device = "cpu"

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))


# Replay Memory 
class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

# Q-network: factorized outputs (n_assets heads, n_actions each)
class QNet(nn.Module):
    def __init__(self, n_observations, n_assets, n_actions_per_asset, hidden=128):
        super(QNet, self).__init__()
        self.n_assets = n_assets
        self.n_actions_per_asset = n_actions_per_asset

        self.fc1 = nn.Linear(n_observations, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_assets * n_actions_per_asset)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        q = self.out(x)
        q = q.view(-1, self.n_assets, self.n_actions_per_asset)
        return q

# Agent
class DQNAgent:
    def __init__(self,obs_dim,n_assets,n_actions_per_asset=5,lr=1e-3,gamma=0.99,batch_size=64,replay_capacity=50000,eps_start=1.0,
        eps_end=0.05,eps_decay=20000,target_update_steps=1000,device="cpu",print_every = 50):

        self.obs_dim = obs_dim
        self.n_assets = n_assets
        self.n_actions_per_asset = n_actions_per_asset
        self.device = device
        self.policy_net = QNet(obs_dim, n_assets, n_actions_per_asset).to(device)
        self.target_net = QNet(obs_dim, n_assets, n_actions_per_asset).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay = ReplayMemory(replay_capacity)
        self.batch_size = batch_size
        self.gamma = gamma

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.step_count = 0
        self.target_update_steps = target_update_steps
        self.print_every = print_every
        self.action_values = np.array([-2, -1, 0, 1, 2], dtype=np.int32)

    def select_action_vector(self, obs, greedy=False):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        q = self.policy_net(obs_t).detach().cpu().numpy()[0]
        action = np.zeros(self.n_assets, dtype=np.int32)
        eps = self.eps_end + (self.eps_start - self.eps_end) * math.exp(-self.step_count / self.eps_decay)
        self.step_count += 1
        for i in range(self.n_assets):
            if (not greedy) and (random.random() < eps):
                idx = random.randrange(self.n_actions_per_asset)
            else:
                idx = int(np.argmax(q[i]))
            action[i] = self.action_values[idx]
        return action

    def store_transition(self, state, action_vec, next_state, reward, done):
        self.replay.push(state.copy(), action_vec.copy(), next_state.copy(), float(reward), bool(done))

    def optimize_model(self):
        if len(self.replay) < self.batch_size:
            return None

        transitions = self.replay.sample(self.batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.tensor(np.stack(batch.state), dtype=torch.float32, device=self.device)
        next_state_batch = torch.tensor(np.stack(batch.next_state), dtype=torch.float32, device=self.device)
        action_batch = torch.tensor(np.stack(batch.action), dtype=torch.int64, device=self.device)
        reward_batch = torch.tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        done_batch = torch.tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.policy_net(state_batch)

        action_index_map = {val: idx for idx, val in enumerate(self.action_values)}
        action_indices = torch.tensor([[action_index_map[int(a)] for a in row] for row in action_batch.cpu().numpy()],
                                      device=self.device, dtype=torch.long)
        action_indices = action_indices.unsqueeze(-1)
        chosen_q = q_values.gather(2, action_indices).squeeze(2)

        with torch.no_grad():
            next_q_target = self.target_net(next_state_batch)
            max_next_q_per_asset = next_q_target.max(dim=2)[0]
            target_q_per_asset = reward_batch + (1.0 - done_batch) * (self.gamma * max_next_q_per_asset)

        loss = F.smooth_l1_loss(chosen_q, target_q_per_asset)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        if self.step_count % self.target_update_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()

    def save(self, fname):
        torch.save(self.policy_net.state_dict(), fname)

    def load(self, fname):
        self.policy_net.load_state_dict(torch.load(fname, map_location=self.device))
        self.target_net.load_state_dict(self.policy_net.state_dict())


# Wealth computation
def obs_to_wealth(obs, n_assets):
    cash = float(obs[0])
    prices = np.array(obs[1:1 + n_assets], dtype=float)
    holdings = np.array(obs[1 + n_assets:1 + n_assets + n_assets], dtype=float)
    return cash + float(np.dot(prices, holdings))


# Training loop
def train_agent(env, objective='terminal', num_episodes=4000, max_steps=10, **kwargs):
    tmp_obs = env.reset()
    obs_dim = len(tmp_obs)
    n_assets = env.num_assets

    agent = DQNAgent(obs_dim, n_assets, **kwargs)

    losses = []
    episode_final_wealth = []
    t0 = time.time()
    for ep in range(1, num_episodes + 1):
        env.seed(seed + ep)
        obs = env.reset()
        total_reward = 0.0

        for t in range(max_steps):
            action = agent.select_action_vector(obs, greedy=False)
            obs_next, env_reward, done, info = env.step(action.astype(np.int32))
            reward = float(env_reward if objective == 'terminal' else obs_to_wealth(obs_next, n_assets))
            agent.store_transition(obs, action, obs_next, reward, done)
            loss = agent.optimize_model()
            if loss is not None:
                losses.append(loss)
            obs = obs_next
            total_reward += reward
            if done:
                break

        final_wealth = obs_to_wealth(obs, n_assets)
        episode_final_wealth.append(final_wealth)

        if (ep % kwargs.get("print_every", 100)) == 0 or ep == 1:
            avg_last = np.mean(episode_final_wealth[-kwargs.get("print_every", 100):])
            print(f"[{objective}] Ep {ep}/{num_episodes}  avg_last: {avg_last:.3f}")

    t1 = time.time()
    print(f"[{objective}] Training done in {t1 - t0:.1f}s.")
    return agent, losses


# Evaluation
def evaluate_agent(agent, env, seeds=100, max_steps=10):
    n_assets = env.num_assets
    wealth_matrix = np.zeros((seeds, max_steps), dtype=float)

    for s in range(seeds):
        env.seed(seed + 1000 + s)
        obs = env.reset()
        for t in range(max_steps):
            action = agent.select_action_vector(obs, greedy=True)
            obs, env_reward, done, info = env.step(action.astype(np.int32))
            wealth_matrix[s, t] = obs_to_wealth(obs, n_assets)
            if done:
                if t < max_steps - 1:
                    wealth_matrix[s, t + 1:] = wealth_matrix[s, t]
                break

    mean_wealth = wealth_matrix.mean(axis=0)
    std_wealth = wealth_matrix.std(axis=0)
    return wealth_matrix, mean_wealth, std_wealth, mean_wealth[-1], std_wealth[-1], mean_wealth[-1] / (std_wealth[-1] + 1e-8)


# Plot helpers
def plot_mean_std(mean_wealth, std_wealth, title, fname):
    steps = np.arange(1, len(mean_wealth) + 1)
    plt.figure(figsize=(8, 5))
    plt.plot(steps, mean_wealth, label='Mean wealth')
    plt.fill_between(steps, mean_wealth - std_wealth, mean_wealth + std_wealth, alpha=0.25, label='Std dev')
    plt.xlabel('Timestep')
    plt.ylabel('Portfolio wealth')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(fname)
    plt.close()


def plot_loss_curve(losses, title, fname, window=50):
    plt.figure(figsize=(8, 4))
    plt.plot(losses, label='Loss', alpha=0.5)

    # Compute moving average
    if len(losses) >= window:
        moving_avg = np.convolve(losses, np.ones(window) / window, mode='valid')
        plt.plot(
            np.arange(window - 1, len(losses)),
            moving_avg,
            label=f'Moving Avg ({window})',
            color='orange',
            linewidth=2
        )

    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.savefig(fname)
    plt.close()


# ------------------------- Main -------------------------
if __name__ == "__main__":
    start_time = time.time()

    env = DiscretePortfolioOptEnv()
    max_steps = env.step_limit

    NUM_EPISODES = 2000
    HYPERPARAMS = dict(
        n_actions_per_asset=5,
        batch_size=128,
        replay_capacity=50000,
        lr=1e-4,
        gamma=0.99,
        eps_start=1.0,
        eps_end=0.05,
        eps_decay=20000,
        target_update_steps=1000,
        device=device,
        print_every=250
    )

    results = {}
    for objective in ['terminal', 'cumulative']:
        print("\n========================================")
        print(f"Training for objective: {objective}")
        agent, losses = train_agent(env, objective=objective, num_episodes=NUM_EPISODES, max_steps=max_steps, **HYPERPARAMS)

        # ---- Save model ----
        model_path = f"models/dqn_factorized_{objective}.pt"
        agent.save(model_path)
        print(f"Saved model to {model_path}")

        # ---- Evaluate ----
        wealth_mat, mean_w, std_w, final_mean, final_std, ratio = evaluate_agent(agent, env, seeds=100, max_steps=max_steps)

        # ---- Save plots ----
        plot_mean_std(mean_w, std_w, f"Wealth progression ({objective})", f"plots/wealth_{objective}.png")
        plot_loss_curve(losses, f"Loss curve ({objective})", f"plots/loss_{objective}.png")

        # ---- Save JSON results ----
        results[objective] = {
            'mean_trajectory': mean_w.tolist(),
            'std_trajectory': std_w.tolist(),
            'final_mean': float(final_mean),
            'final_std': float(final_std),
            'ratio': float(ratio),
        }

    with open("evaluation/dqn_results_summary.json", "w") as f:
        json.dump(results, f, indent=4)
    print("Saved results to evaluation/dqn_results_summary.json")
