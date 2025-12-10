import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import imageio
from collections import namedtuple, deque
from env import TreasureHunt_v2

# ------------------ Transition & HER Replay ------------------
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'goal'))

class HindsightReplayBuffer:
    def __init__(self, capacity, her_k=4):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.her_k = her_k  # number of HER samples per episode

    def push_episode(self, episode_transitions):
        self.buffer.extend(episode_transitions)
        # HER: sample future states as new goals
        for i, trans in enumerate(episode_transitions):
            state, action, reward, next_state, done, goal = trans
            future_indices = np.random.choice(
                range(i, len(episode_transitions)), 
                size=min(self.her_k, len(episode_transitions) - i),
                replace=False
            )
            for j in future_indices:
                future_state = episode_transitions[j].next_state
                new_goal = np.copy(future_state[3])
                achieved = next_state[3]
                new_reward = 1.0 if np.array_equal(achieved, new_goal) else 0.0
                new_done = bool(new_reward)
                new_transition = Transition(state, action, new_reward, next_state, new_done, new_goal)
                self.buffer.append(new_transition)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


# ------------------ Network ------------------
class DQN(nn.Module):
    def __init__(self, input_channels=5, num_actions=4):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, 3, padding=1)
        self.conv2 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 64, 3, stride=2, padding=1)
        with torch.no_grad():
            dummy = torch.zeros(1, input_channels, 10, 10)
            flat_size = self._forward_conv(dummy).view(1, -1).shape[1]
        self.fc1 = nn.Linear(flat_size, 64)
        self.out = nn.Linear(64, num_actions)

    def _forward_conv(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.out(x)


# ------------------ Evaluation ------------------
def evaluate_policy(env, policy_net, device, num_episodes=100, max_steps=500):
    policy_net.eval()
    rewards = []

    for _ in range(num_episodes):
        state = env.reset()
        goal = np.copy(state[3])  # treasure layer as goal
        total_reward = 0
        for _ in range(max_steps):
            input_state = np.concatenate((state, goal[np.newaxis, :, :]), axis=0)
            with torch.no_grad():
                s = torch.from_numpy(input_state.astype(np.float32)).unsqueeze(0).to(device)
                action = policy_net(s).argmax(1).item()
            nstate, reward = env.step(action)
            total_reward += reward
            state = nstate
            if bool(state[3, 9, 9]):  # reached fort
                break
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards), rewards


def evaluate_random_policy(env, num_episodes=50, max_steps=500):
    rewards = []
    for _ in range(num_episodes):
        state = env.reset()
        total_reward = 0
        for _ in range(max_steps):
            action = random.randint(0, 3)
            nstate, reward = env.step(action)
            total_reward += reward
            state = nstate
            if bool(state[3, 9, 9]):
                break
        rewards.append(total_reward)
    return np.mean(rewards), np.std(rewards), rewards


# ------------------ Training ------------------
def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    env = TreasureHunt_v2()
    memory = HindsightReplayBuffer(50000, her_k=4)

    policy_net = DQN().to(device)
    target_net = DQN().to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.Adam(policy_net.parameters(), lr=1e-3)

    gamma = 0.99
    eps_start, eps_end, eps_decay = 1.0, 0.05, 0.995
    batch_size = 64
    num_episodes = 5000
    max_steps = 500

    rewards, losses = [],[]
    total_steps = 0

    for ep in range(1, num_episodes + 1):
        state = env.reset()
        goal = np.copy(state[3])
        eps = max(eps_start * (eps_decay ** ep), eps_end)
        episode_transitions = []
        total_reward = 0

        for _ in range(max_steps):
            input_state = np.concatenate((state, goal[np.newaxis, :, :]), axis=0)
            if random.random() < eps:
                action = random.randint(0, 3)
            else:
                with torch.no_grad():
                    s = torch.from_numpy(input_state.astype(np.float32)).unsqueeze(0).to(device)
                    action = policy_net(s).argmax(1).item()

            next_state, reward = env.step(action)
            done = bool(next_state[3, 9, 9])
            episode_transitions.append(Transition(state, action, reward, next_state, done, goal))
            state = next_state
            total_reward += reward
            total_steps += 1
            if done:
                break

        memory.push_episode(episode_transitions)
        rewards.append(total_reward)

        if len(memory) >= batch_size:
            batch = memory.sample(batch_size)
            s_batch = np.stack(batch.state)
            g_batch = np.stack(batch.goal)
            sg_batch = np.concatenate((s_batch, g_batch[:, np.newaxis, :, :]), axis=1)
            ns_batch = np.stack(batch.next_state)
            nsg_batch = np.concatenate((ns_batch, g_batch[:, np.newaxis, :, :]), axis=1)

            s = torch.from_numpy(sg_batch).float().to(device)
            ns = torch.from_numpy(nsg_batch).float().to(device)
            a = torch.tensor(batch.action, dtype=torch.long, device=device).unsqueeze(1)
            r = torch.tensor(batch.reward, dtype=torch.float32, device=device).unsqueeze(1)
            d = torch.tensor(batch.done, dtype=torch.float32, device=device).unsqueeze(1)

            q_vals = policy_net(s).gather(1, a)
            next_q = target_net(ns).max(1, keepdim=True)[0].detach()
            target = r + gamma * next_q * (1 - d)
            loss = F.smooth_l1_loss(q_vals, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        if total_steps % 50 == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if ep % 10 == 0:
            print(f"Ep {ep}: avg_reward={np.mean(rewards[-10:]):.3f}, eps={eps:.3f}, buffer={len(memory)}")

    # Save model
    os.makedirs('model', exist_ok=True)
    torch.save(policy_net.state_dict(), 'model/dqn_her.pt')

    # Evaluate
    mean_t, std_t, eval_t = evaluate_policy(env, policy_net, device)
    mean_r, std_r, eval_r = evaluate_random_policy(env)

    print(f"\nTrained Policy — Mean: {mean_t:.3f} ± {std_t:.3f}")
    print(f"Random Policy  — Mean: {mean_r:.3f} ± {std_r:.3f}")

    save_plots(rewards, losses, eval_t, eval_r)
    #save_gifs(env, policy_net, device)


# ------------------ Plot & GIF ------------------
def save_plots(train_rewards, losses, eval_trained, eval_random):
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    os.makedirs('plots', exist_ok=True)

    # ------------------ Training Plots ------------------
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_rewards, alpha=0.3)
    if len(train_rewards) >= 50:
        ma = np.convolve(train_rewards, np.ones(50)/50, mode='valid')
        plt.plot(range(49, len(train_rewards)), ma, lw=2)
    plt.title("Training Rewards")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(losses, alpha=0.6)
    if len(losses) >= 100:
        ma = np.convolve(losses, np.ones(100)/100, mode='valid')
        plt.plot(range(99, len(losses)), ma, lw=2, color='r')
    plt.title("Training Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('plots/training.png')
    plt.close()

    # ------------------ Evaluation: Trained Policy ------------------
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(eval_trained, bins=20, edgecolor='black', alpha=0.7)
    plt.axvline(np.mean(eval_trained), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(eval_trained):.2f}')
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title('Trained Policy: Reward Distribution (100 episodes)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(eval_trained, marker='o', linestyle='-', alpha=0.6)
    plt.axhline(np.mean(eval_trained), color='red', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(eval_trained):.2f}')
    plt.fill_between(range(len(eval_trained)),
                     np.mean(eval_trained) - np.std(eval_trained),
                     np.mean(eval_trained) + np.std(eval_trained),
                     alpha=0.2, color='red',
                     label=f'±1 Std: {np.std(eval_trained):.2f}')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Reward')
    plt.title('Trained Policy: Episode Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/evaluation_trained.png', dpi=150)
    plt.close()

    # ------------------ Evaluation: Random Policy ------------------
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.hist(eval_random, bins=20, edgecolor='black', alpha=0.7, color='blue')
    plt.axvline(np.mean(eval_random), color='blue', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(eval_random):.2f}')
    plt.xlabel('Episode Reward')
    plt.ylabel('Frequency')
    plt.title('Random Policy: Reward Distribution (100 episodes)')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(eval_random, marker='o', linestyle='-', alpha=0.6, color='blue')
    plt.axhline(np.mean(eval_random), color='blue', linestyle='--', linewidth=2,
                label=f'Mean: {np.mean(eval_random):.2f}')
    plt.fill_between(range(len(eval_random)),
                     np.mean(eval_random) - np.std(eval_random),
                     np.mean(eval_random) + np.std(eval_random),
                     alpha=0.2, color='blue',
                     label=f'±1 Std: {np.std(eval_random):.2f}')
    plt.xlabel('Evaluation Episode')
    plt.ylabel('Reward')
    plt.title('Random Policy: Episode Rewards')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plots/evaluation_random.png', dpi=150)
    plt.close()


def save_gifs(env, policy_net, device):
    os.makedirs('gifs', exist_ok=True)
    policy_net.eval()

    # ========== Trained Agent ==========
    state = env.reset()
    goal = np.copy(state[3])  # treasure layer as goal
    frames = [env.render(state, return_image=True)]

    for _ in range(500):
        input_state = np.concatenate((state, goal[np.newaxis, :, :]), axis=0)
        with torch.no_grad():
            s = torch.from_numpy(input_state.astype(np.float32)).unsqueeze(0).to(device)
            action = policy_net(s).argmax(1).item()
        next_state, reward = env.step(action)
        frames.append(env.render(next_state, return_image=True))
        state = next_state
        if bool(state[3, 9, 9]):  # reached fort
            break

    imageio.mimsave('gifs/trained_treasurehunt.gif',[np.array(f) for f in frames], duration=0.2)

    # ========== Random Agent ==========
    state = env.reset()
    goal = np.copy(state[3])
    frames = [env.render(state, return_image=True)]

    for _ in range(500):
        action = random.randint(0, 3)
        next_state, reward = env.step(action)
        frames.append(env.render(next_state, return_image=True))
        state = next_state
        if bool(state[3, 9, 9]):
            break

    imageio.mimsave('gifs/random_treasurehunt.gif',[np.array(f) for f in frames], duration=0.2)

if __name__ == "__main__":
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
    train()
