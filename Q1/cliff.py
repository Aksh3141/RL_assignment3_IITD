import gymnasium as gym
import numpy as np
import pygame
import random
import os
import imageio
import torch
from dataclasses import dataclass
from gymnasium import spaces

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

SEED = 509
set_seed(SEED)

class MultiGoalCliffWalkingEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, render_mode=None, height=10, width=20, cell_size=60, train = True):
        self.height = height
        self.width = width
        self.cell_size = cell_size

        # Actions: up, right, down, left
        self.action_space = spaces.Discrete(4)

        self.observation_space = spaces.Discrete(self.height * self.width * 4)

        # Rewards and penalties
        self.cliff_penalty = -50         
        self.step_penalty = -1
        self.whisky_penalty = -1
        self.safe_goal_reward = 40
        self.risky_goal_reward = 200

        if train:
            self.slip_probability = 0.1
            self.wine_cels = [(3, 5), (6, 10), (7, 15)]
        
        else:
            self.slip_probability = 0.5
            self.wine_cels = [(6, 5), (7, 10), (7, 17)]


        # Special cells
        self.cliff_row = self.height - 1
        self.slippery_row = self.height - 2

        # Goals and checkpoints
        self.safe_goal = (self.height - 1, self.width  // 2)
        self.risky_goal = (self.height - 1, self.width - 1)
        self.checkpointA = (self.height // 2, self.width // 2)
        self.checkpointB = (self.height - 3, self.width - 2)

        self.render_mode = render_mode

        self.window_width = self.width * self.cell_size
        self.window_height = self.height * self.cell_size
        self.window = None

        self.clock = None

        self.whisky_img = pygame.image.load("./images/wine_glass.png")
        self.whisky_img = pygame.transform.scale(self.whisky_img, (self.cell_size, self.cell_size))

        self.safe_treasure = pygame.image.load("./images/coin.png")
        self.safe_treasure = pygame.transform.scale(self.safe_treasure, (self.cell_size, self.cell_size))

        self.risky_treasure = pygame.image.load("./images/coins.png")
        self.risky_treasure = pygame.transform.scale(self.risky_treasure, (self.cell_size, self.cell_size))

        
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = [self.height - 1, 0]
        self.checkA = False
        self.checkB = False
        return self._get_obs(), {}

    def _state_to_index(self, r, c, checkA, checkB):
        base = r * self.width + c
        checkpoint_bits = (int(checkA) << 1) | int(checkB)  # 0–3
        return base * 4 + checkpoint_bits

    def _get_obs(self):
        r, c = self.agent_pos
        return self._state_to_index(r, c, self.checkA, self.checkB)

    def step(self, action):
        r, c = self.agent_pos

        if (r, c) in self.wine_cels and np.random.rand() < 0.5:
            action = np.random.choice([0, 1, 2, 3])

        if r == self.slippery_row and np.random.rand() < self.slip_probability:
            action = np.random.choice([0, 1, 2, 3])

        # Apply movement
        if action == 0 and r > 0:  
            r -= 1
        elif action == 1 and c < self.width - 1:  
            c += 1
        elif action == 2 and r < self.height - 1: 
            r += 1
        elif action == 3 and c > 0:  
            c -= 1

        self.agent_pos = [r, c]
        reward = self.step_penalty
        terminated = False
        goal = None

        # Whisky penalty
        if (r, c) in self.wine_cels:
            reward += self.whisky_penalty

        # Checkpoints
        if (r, c) == self.checkpointA:
            self.checkA = True
        if (r, c) == self.checkpointB:
            self.checkB = True

        # Cliff
        if r == self.cliff_row and c not in [0, self.width - 1]:
            reward = self.cliff_penalty
            terminated = True
            goal = "cliff"

        # Safe goal
        if (r, c) == self.safe_goal and self.checkA:
            reward = self.safe_goal_reward
            terminated = True
            goal = "safe"

        # Risky goal
        if (r, c) == self.risky_goal and self.checkA and self.checkB:
            reward = self.risky_goal_reward
            terminated = True
            goal = "risky"

        return self._get_obs(), reward, terminated, False, {"goal": goal}


    def render(self):
        if self.window is None:
            pygame.init()
            self.window = pygame.display.set_mode((self.window_width, self.window_height))
            self.clock = pygame.time.Clock()

        # Fill background
        self.window.fill((255, 255, 255)) 

        for r in range(self.height):
            for c in range(self.width):
                rect = pygame.Rect(c * self.cell_size, r * self.cell_size,
                                self.cell_size, self.cell_size)

                # Base color for cell
                color = (43, 134, 224)  

                if r == self.cliff_row:
                    if c not in [0, self.width - 1]:
                        color = (255, 180, 180) 
                    else:
                        color = (43, 134, 224)  
                
                elif r == self.slippery_row:
                    color = (3, 235, 252)


                pygame.draw.rect(self.window, color, rect)
                pygame.draw.rect(self.window, (0, 0, 0), rect, 1)  

                if (r, c) == self.safe_goal:
                    self.window.blit(self.safe_treasure, (c * self.cell_size, r * self.cell_size))

                if (r, c) == self.risky_goal:
                    self.window.blit(self.risky_treasure, (c * self.cell_size, r * self.cell_size))

                if (r, c) in self.wine_cels:
                    self.window.blit(self.whisky_img, (c * self.cell_size, r * self.cell_size))

                if (r, c) == self.checkpointA:
                    color_chk = (135, 206, 250) if not self.checkA else (0, 0, 139)  
                    center = (c * self.cell_size + self.cell_size // 2, r * self.cell_size + self.cell_size // 2)
                    radius = self.cell_size // 3
                    pygame.draw.circle(self.window, color_chk, center, radius)

                if (r, c) == self.checkpointB:
                    color_chk = (144, 238, 144) if not self.checkB else (0, 100, 0)  
                    center = (c * self.cell_size + self.cell_size // 2, r * self.cell_size + self.cell_size // 2)
                    radius = self.cell_size // 3
                    pygame.draw.circle(self.window, color_chk, center, radius)

        ar, ac = self.agent_pos
        pygame.draw.circle(self.window, (255, 165, 0),
                        (ac * self.cell_size + self.cell_size // 2,
                            ar * self.cell_size + self.cell_size // 2),
                        self.cell_size // 3)

        pygame.display.update()

        if self.render_mode == "human":
            self.clock.tick(self.metadata["render_fps"])

        if self.render_mode == "rgb_array":
            canvas = pygame.surfarray.array3d(self.window)
            return np.transpose(canvas, (1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.quit()
            self.window = None

