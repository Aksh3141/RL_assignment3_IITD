import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
import json
import os
from matplotlib import animation
import matplotlib.pyplot as plt
from collections import namedtuple

MODEL_DIR = 'models'
EVAL_DIR = 'evaluation'
PLOT_DIR = 'plots'
GIF_DIR = 'gifs'
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(EVAL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)
os.makedirs(GIF_DIR, exist_ok=True)

def set_seed(seed=42):
    """Set random seed for reproducibility."""
    import torch, random, numpy as np, os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

# Hyperparameters
GAMMA = 0.99               
LR = 5e-3                  
BATCH_SIZE = 64             
TARGET_UPDATE_FREQ = 1000   
BUFFER_SIZE = 5000          
MAX_EPISODES = 50000        
EVAL_FREQ = 10              
MAX_STEPS_PER_EPISODE = 1000 
TRAINING_STEPS_PER_BATCH = 1024 

# Epsilon-greedy exploration schedule
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 0.995

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.Transition = namedtuple("Transition", 
            ["state", "action", "reward", "next_state", "next_action", "done"])

    def push(self, state, action, reward, next_state, next_action, done):
        """Add new experience with max priority (so it's sampled soon)."""
        max_prio = self.priorities.max() if self.buffer else 1.0
        transition = self.Transition(state, action, reward, next_state, next_action, done)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size, beta=0.4):
        """Sample a batch of experiences with importance sampling."""
        if len(self.buffer) == self.capacity:
            prios = self.priorities
        else:
            prios = self.priorities[:self.pos]

        probs = prios ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]

        # Compute weights
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)

        batch = self.Transition(*zip(*samples))
        return batch, indices, torch.tensor(weights, dtype=torch.float32)

    def update_priorities(self, batch_indices, batch_priorities):
        for idx, prio in zip(batch_indices, batch_priorities):
            self.priorities[idx] = prio

    def __len__(self):
        return len(self.buffer)

    def clear(self):
        self.buffer = []
        self.priorities = np.zeros((self.capacity,), dtype=np.float32)
        self.pos = 0



#Q-Network Definition
class QNetwork(nn.Module):
    
    def __init__(self, state_size, action_size, seed):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Deep SARSA Agent
class DeepSarsaAgent:
    def __init__(self, state_size, action_size, seed):
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.steps = 0 

        # Q-Network (local) and Target Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed).to(DEVICE)
        self.qnetwork_target = QNetwork(state_size, action_size, seed).to(DEVICE)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Temporary Buffer 
        self.buffer = PrioritizedReplayBuffer(BUFFER_SIZE)


    def step(self, state, action, reward, next_state, next_action, done):
        self.buffer.push(state, action, reward, next_state, next_action, done)


    def act(self, state, eps):
        #Returns actions for given state using epsilon-greedy policy.
        state = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, beta=0.4):
        if len(self.buffer) < BATCH_SIZE:
            return

        # Sample experiences from prioritized buffer
        experiences, indices, weights = self.buffer.sample(BATCH_SIZE, beta)
        
        states = torch.tensor(np.vstack(experiences.state), dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(np.vstack(experiences.action), dtype=torch.long).to(DEVICE)
        rewards = torch.tensor(np.vstack(experiences.reward), dtype=torch.float32).to(DEVICE)
        next_states = torch.tensor(np.vstack(experiences.next_state), dtype=torch.float32).to(DEVICE)
        next_actions = torch.tensor(np.vstack(experiences.next_action), dtype=torch.long).to(DEVICE)
        dones = torch.tensor(np.vstack(experiences.done).astype(np.uint8), dtype=torch.float32).to(DEVICE)
        weights = weights.unsqueeze(1).to(DEVICE)

        # Compute Q values and targets
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        with torch.no_grad():
            Q_targets_next = self.qnetwork_target(next_states).gather(1, next_actions)
        Q_targets = rewards + (GAMMA * Q_targets_next * (1 - dones))

        # Compute TD error and loss
        td_errors = (Q_expected - Q_targets).detach().cpu().numpy()
        loss = (weights * (Q_expected - Q_targets).pow(2)).mean()

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update priorities in buffer
        new_priorities = np.abs(td_errors) + 1e-5
        self.buffer.update_priorities(indices, new_priorities)

        # Soft update of target network
        self.soft_update(self.qnetwork_local, self.qnetwork_target, 1.0 / TARGET_UPDATE_FREQ)


    def discard_buffer(self):
        """Completely discard all collected transitions."""
        self.buffer.clear()

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


# Training Function
def train_deep_sarsa(env, agent, n_episodes):
    scores = deque(maxlen=100)
    scores_list = []
    best_score = -np.inf
    eps = EPS_START

    for i_episode in range(1, n_episodes + 1):

        eps = max(EPS_END,eps*EPS_DECAY)

        # Training Loop
        if i_episode % 10 == 1:
            agent.discard_buffer()
        state, _ = env.reset()
        current_step = 0

        while current_step < TRAINING_STEPS_PER_BATCH:
            action = agent.act(state, eps)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_action = agent.act(next_state, eps)

            agent.step(state, action, reward, next_state, next_action, done)
            state = next_state
            current_step += 1

            if done:
                state, _ = env.reset()
                # Break if the first episode finished before reaching the batch size
                if current_step < BATCH_SIZE:
                    break
    
        agent.learn()

        # Evaluation & Model Saving
        if i_episode % EVAL_FREQ == 0:
            avg_score = evaluate_agent(env, agent.qnetwork_local)
            scores_list.append(avg_score)
            print(f'\rEpisode {i_episode}\tAverage Score (10 eps): {avg_score:.2f}\tEpsilon: {eps:.2f}', end="")
            
            # Save the best model
            if avg_score > best_score:
                best_score = avg_score
                print(f"\nSaving new best model with score {best_score:.2f}...")
                torch.save(agent.qnetwork_local.state_dict(), os.path.join(MODEL_DIR, 'best_deep_sarsa.pt'))

        if i_episode % 100 == 0:
            print(f'\rEpisode {i_episode}\tAverage Score (10 eps): {avg_score:.2f}\tEpsilon: {eps:.2f}')

        # After training, plot reward curve
    plt.figure(figsize=(8, 5))
    plt.plot(np.arange(len(scores_list)) * EVAL_FREQ, scores_list, label='Avg Reward (Eval)')
    plt.xlabel('Episode')
    plt.ylabel('Average Reward')
    plt.title('Deep SARSA Training Performance')
    plt.legend()
    plt.grid(True)
    plot_path = os.path.join(PLOT_DIR, 'reward_plot.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"\nReward plot saved to {plot_path}")

    return scores_list


# Evaluation Functions
def evaluate_agent(env, qnetwork, n_episodes=10):
    """Evaluates the agent's performance over n_episodes."""
    rewards = []
    qnetwork.eval()
    for _ in range(n_episodes):
        state, _ = env.reset()
        score = 0
        done = False
        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action = qnetwork(state_tensor).argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
        rewards.append(score)
    qnetwork.train()
    return np.mean(rewards)

def final_evaluation(env, model_path, n_episodes=100):
    """Loads the best model and performs the final 100-episode evaluation."""
    print(f"\n--- Final Evaluation (100 Episodes) ---")
    
    # Load the best model
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    qnetwork = QNetwork(state_size, action_size, seed=0).to(DEVICE)
    qnetwork.load_state_dict(torch.load(model_path, map_location=DEVICE))
    qnetwork.eval()
    best_reward = -float("inf") 
    rewards = []
    for i_episode in range(1, n_episodes + 1):
        state, _ = env.reset()
        score = 0
        done = False
        while not done:
            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                action = qnetwork(state_tensor).argmax().item()
            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            score += reward
        rewards.append(score)
        print(f"\rEpisode {i_episode}/{n_episodes}\tScore: {score:.2f}", end="")
        if rewards[-1] > best_reward:
            return_qnetwork = qnetwork
            best_reward = rewards[-1]
    mean_reward = np.mean(rewards)
    std_reward = np.std(rewards)

    # Report results
    results = {
        "mean_reward": mean_reward,
        "std_reward": std_reward
    }

    # Save results to JSON
    with open(os.path.join(EVAL_DIR, 'lunarlandar_evaluation_results.json'), 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\nMean Reward: {mean_reward:.2f}, Std Deviation: {std_reward:.2f}")
    print(f"Results saved to {os.path.join(EVAL_DIR, 'lunarlandar_evaluation_results.json')}")

    return return_qnetwork


# --- 6. GIF Creation Function ---
def create_gif(env, qnetwork, filename):
    """Renders an episode and creates a GIF."""
    print(f"\n--- Creating GIF ---")
    
    state, _ = env.reset()
    done = False
    frames = []

    while not done:
        frames.append(env.render())
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = qnetwork(state_tensor).argmax().item()
        state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

    # Use matplotlib to save the frames as a GIF
    fig = plt.figure()
    plt.axis('off')
    
    if not frames:
        print("Error: No frames collected for GIF.")
        return

    patch = plt.imshow(frames[0])
    
    def animate(i):
        patch.set_data(frames[i])
    
    ani = animation.FuncAnimation(fig, animate, frames=len(frames), interval=50) 
    ani.save(filename, writer='pillow', fps=20)
    plt.close(fig)
    print(f"GIF saved to {filename}")


# Main
if __name__ == '__main__':
    # Initialize environments
    SEED = 42
    set_seed(SEED)
    env_train = gym.make('LunarLander-v3')
    env_render = gym.make('LunarLander-v3', render_mode='rgb_array')
    env_render.reset(seed=SEED)
    env_render.action_space.seed(SEED)
    env_render.observation_space.seed(SEED)
    state_size = env_train.observation_space.shape[0]
    action_size = env_train.action_space.n
    
    # Initialize agent
    agent = DeepSarsaAgent(state_size, action_size, seed=0)

    # Start Training
    print("Starting Deep SARSA Training...")
    train_deep_sarsa(env_train, agent, n_episodes=MAX_EPISODES)
    print("\nTraining Complete.")

    
    BEST_MODEL_PATH = os.path.join(MODEL_DIR, 'best_deep_sarsa.pt')
    best_qnetwork = final_evaluation(env_render, BEST_MODEL_PATH, n_episodes=100)

    # Create GIF
    create_gif(env_render, best_qnetwork, os.path.join(GIF_DIR, 'lunarlandar.gif'))
    env_train.close()
    env_render.close()