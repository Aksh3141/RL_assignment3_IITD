import os
import json
import random
import numpy as np
from collections import deque, namedtuple
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


# Replay buffer with priorities
Transition = namedtuple("Transition", ("state", "action", "reward", "next_state", "done"))

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    import torch, random, numpy as np, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos = 0

    def push(self, *args):
        max_prio = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.pos] = Transition(*args)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]

        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

        batch = Transition(*zip(*samples))
        states = torch.tensor(batch.state, dtype=torch.long)
        actions = torch.tensor(batch.action, dtype=torch.long).unsqueeze(1)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).unsqueeze(1)
        next_states = torch.tensor(batch.next_state, dtype=torch.long)
        dones = torch.tensor(batch.done, dtype=torch.float32).unsqueeze(1)
        return states, actions, rewards, next_states, dones, indices, weights

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)


# Q-networks

class LinearQNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.fc(x)

class NonlinearQNet(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden , hidden // 2),
            nn.ReLU(),
            nn.Linear(hidden // 2, output_dim)
        )
    def forward(self, x):
        return self.net(x)


# Helper functions

def states_to_onehot(states, n_states, device):
    return torch.nn.functional.one_hot(states.to(device), num_classes=n_states).float()

def polyak_update(target_net, q_net, tau=0.005):
    with torch.no_grad():
        for target_param, param in zip(target_net.parameters(), q_net.parameters()):
            target_param.data.mul_(1 - tau)
            target_param.data.add_(tau * param.data)

def select_action(q_net, state, n_states, epsilon, device):
    if random.random() < epsilon:
        return random.randint(0, q_net(torch.zeros(1, n_states, device=device)).shape[-1] - 1)
    s = states_to_onehot(torch.tensor([state]), n_states, device)
    with torch.no_grad():
        q_values = q_net(s)
        return int(torch.argmax(q_values, dim=-1).item())


# Training function

def train_dqn(env, n_states, n_actions, model_type, save_path,
              num_episodes=5000, max_steps=1000, batch_size=64, gamma=0.995, lr=5e-4,
              replay_capacity=60000, tau=0.01, epsilon_start=1.0, epsilon_end=0.1,
              epsilon_decay_episodes=2000, alpha=0.4, beta_start=0.6):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {model_type} on {device}")

    if model_type == "linear":
        q_net = LinearQNet(n_states, n_actions).to(device)
    else:
        q_net = NonlinearQNet(n_states, n_actions).to(device)

    target_net = type(q_net)(n_states, n_actions).to(device)
    target_net.load_state_dict(q_net.state_dict())

    optimizer = optim.Adam(q_net.parameters(), lr=lr)
    replay = PrioritizedReplayBuffer(replay_capacity, alpha=alpha)

    best_avg = -float("inf")
    episode_rewards, moving_avgs = [], []

    def epsilon(ep):
        return max(epsilon_end, epsilon_start - (ep / epsilon_decay_episodes) * (epsilon_start - epsilon_end))

    beta = beta_start
    beta_inc = (1.0 - beta_start) / num_episodes

    for ep in range(1, num_episodes + 1):
        state, _ = env.reset()
        total_reward = 0.0
        done = False
        eps = epsilon(ep)

        for step in range(max_steps):
            action = select_action(q_net, state, n_states, eps, device)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay.push(state, action, reward, next_state, float(done))
            state = next_state
            total_reward += reward

            if len(replay) >= batch_size:
                states_b, actions_b, rewards_b, next_states_b, dones_b, indices, weights = replay.sample(batch_size, beta=beta)
                states_b = states_to_onehot(states_b, n_states, device)
                next_states_b = states_to_onehot(next_states_b, n_states, device)

                q_values = q_net(states_b).gather(1, actions_b.to(device))
                with torch.no_grad():
                    next_q = target_net(next_states_b).max(dim=1, keepdim=True)[0]
                    target_q = rewards_b.to(device) + gamma * next_q * (1 - dones_b.to(device))

                td_error = (target_q - q_values).detach().cpu().numpy().flatten()
                loss = (weights.to(device) * (q_values - target_q).pow(2)).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                replay.update_priorities(indices, np.abs(td_error) + 1e-5)

                polyak_update(target_net, q_net, tau)

            if done:
                break

        beta = min(1.0, beta + beta_inc)
        episode_rewards.append(total_reward)
        avg = np.mean(episode_rewards[-50:])
        moving_avgs.append(avg)

        if avg > best_avg and ep >= 50:
            best_avg = avg
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(q_net.state_dict(), save_path)

        if ep % 25 == 0 or ep == 1 or ep == num_episodes:
            print(f"[{model_type}] Ep {ep}/{num_episodes} reward={total_reward:.1f} avg50={avg:.1f} eps={eps:.3f}")

    return episode_rewards, moving_avgs


# Plot helper

def plot_rewards(rewards, moving_avg, path, title):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.figure(figsize=(10,6))
    plt.plot(rewards, alpha=0.3, label="Reward per episode")
    plt.plot(moving_avg, label="Moving average (50)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()


# Evaluation

def evaluate_model(env, n_states, n_actions, model_class, path, episodes=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(n_states, n_actions).to(device)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.eval()

    rewards = []
    for _ in range(episodes):
        s, _ = env.reset()
        done = False
        total = 0
        for t in range(1000):
            inp = states_to_onehot(torch.tensor([s]), n_states, device)
            with torch.no_grad():
                a = int(torch.argmax(model(inp)).item())
            s2, r, term, trunc, _ = env.step(a)
            total += r
            s = s2
            done = term or trunc
            if done:
                break
        rewards.append(total)

    return float(np.mean(rewards)), float(np.std(rewards)), rewards


# Main
def main():
    from cliff import MultiGoalCliffWalkingEnv
    SEED = 42
    set_seed(SEED)

    os.makedirs("models", exist_ok=True)
    os.makedirs("plots", exist_ok=True)
    os.makedirs("evaluation", exist_ok=True)

    # --- TRAIN environments with fixed seed ---
    train_env = MultiGoalCliffWalkingEnv(train=True)
    train_env.reset(seed=SEED)
    train_env.action_space.seed(SEED)
    train_env.observation_space.seed(SEED)

    obs_n = train_env.observation_space.n
    act_n = train_env.action_space.n

    NUM_EPISODES = 5000
    MAX_STEPS = 1000

    # ---- Linear ----
    linear_model_path = "models/best_linear.pt"
    print("=== Training Linear DQN (PER + Polyak) ===")
    linear_rewards, linear_avg = train_dqn(train_env, obs_n, act_n, "linear", linear_model_path,num_episodes=NUM_EPISODES, max_steps=MAX_STEPS)
    plot_rewards(linear_rewards, linear_avg, "plots/cliff average rewards linear.png","Linear DQN (PER + Polyak)")

    # ---- Nonlinear ----
    train_env_nl = MultiGoalCliffWalkingEnv(train=True)
    train_env_nl.reset(seed=SEED + 1)
    train_env_nl.action_space.seed(SEED + 1)
    train_env_nl.observation_space.seed(SEED + 1)

    nonlinear_model_path = "models/best_nonlinear.pt"
    print("=== Training Nonlinear DQN (PER + Polyak) ===")
    nonlinear_rewards, nonlinear_avg = train_dqn(train_env_nl, obs_n, act_n, "nonlinear", nonlinear_model_path,num_episodes=NUM_EPISODES, max_steps=MAX_STEPS)
    plot_rewards(nonlinear_rewards, nonlinear_avg, "plots/cliff average rewards nonlinear.png","Nonlinear DQN (PER + Polyak)")

    # ---- Evaluation ----
    eval_env = MultiGoalCliffWalkingEnv(train=False)
    eval_env.reset(seed=SEED + 2)
    eval_env.action_space.seed(SEED + 2)
    eval_env.observation_space.seed(SEED + 2)

    print("=== Evaluating saved models (100 episodes each) ===")
    lin_mean, lin_std, _ = evaluate_model(eval_env, obs_n, act_n, LinearQNet, linear_model_path)
    non_mean, non_std, _ = evaluate_model(eval_env, obs_n, act_n, NonlinearQNet, nonlinear_model_path)
    results = {
        "linear": {"mean": lin_mean, "std": lin_std},
        "nonlinear": {"mean": non_mean, "std": non_std}
    }

    with open("evaluation/cliff_evaluation_results.json", "w") as f:
        json.dump(results, f, indent=2)

    print("=== Results ===")
    print(json.dumps(results, indent=2))

    # --- Close all envs ---
    train_env.close()
    train_env_nl.close()
    eval_env.close()

if __name__ == "__main__":
    main()
