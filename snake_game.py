import gym
from gym import spaces
import numpy as np
import random
import pygame
import time
from stable_baselines3 import PPO

# ---------------------------
# Global Constants for Rendering and Environment
# ---------------------------
CELL_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20
SCREEN_WIDTH = GRID_WIDTH * CELL_SIZE
SCREEN_HEIGHT = GRID_HEIGHT * CELL_SIZE

# Define Colors (RGB)
BLACK   = (0, 0, 0)
GREEN   = (0, 255, 0)
RED     = (255, 0, 0)
ORANGE  = (255, 165, 0)

# ---------------------------
# Snake Environment Definition
# ---------------------------
class SnakeEnv(gym.Env):
    """
    A simple Snake environment with traps and fruit.
    
    Grid values:
      0: empty cell
      1: snake body
      2: fruit
      3: trap
    """
    metadata = {"render.modes": ["human"]}
    
    def __init__(self):
        super(SnakeEnv, self).__init__()
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_HEIGHT
        
        # The observation is a flattened grid with values 0-3.
        self.observation_space = spaces.Box(low=0, high=3,
                                            shape=(self.grid_width * self.grid_height,),
                                            dtype=np.uint8)
        # Actions: 0 = Up, 1 = Down, 2 = Left, 3 = Right.
        self.action_space = spaces.Discrete(4)
        self.reset()

    def reset(self):
        self.snake = [(self.grid_width // 2, self.grid_height // 2)]
        self.direction = (1, 0)  # Start moving right.
        self.traps = []          # Initialize traps before spawning fruit.
        self.fruit = self._spawn_fruit()
        self.step_count = 0
        self.done = False
        # For reward shaping: track the Manhattan distance from snake head to fruit.
        self.previous_distance = abs(self.snake[0][0] - self.fruit[0]) + abs(self.snake[0][1] - self.fruit[1])
        return self._get_observation()

    def _spawn_fruit(self):
        while True:
            pos = (random.randint(0, self.grid_width - 1),
                   random.randint(0, self.grid_height - 1))
            if pos not in self.snake and pos not in self.traps:
                return pos

    def _spawn_trap(self):
        while True:
            pos = (random.randint(0, self.grid_width - 1),
                   random.randint(0, self.grid_height - 1))
            if pos not in self.snake and pos != self.fruit and pos not in self.traps:
                return pos

    def _get_observation(self):
        grid = np.zeros((self.grid_width, self.grid_height), dtype=np.uint8)
        # Mark the snake.
        for pos in self.snake:
            grid[pos[0], pos[1]] = 1
        # Mark the fruit.
        fx, fy = self.fruit
        grid[fx, fy] = 2
        # Mark the traps.
        for trap in self.traps:
            grid[trap[0], trap[1]] = 3
        return grid.flatten()

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, {}
        
        # Map the action to a new direction.
        if action == 0:
            new_direction = (0, -1)   # Up
        elif action == 1:
            new_direction = (0, 1)    # Down
        elif action == 2:
            new_direction = (-1, 0)   # Left
        elif action == 3:
            new_direction = (1, 0)    # Right
        else:
            new_direction = self.direction
        
        # Prevent reversal if snake has more than one segment.
        if len(self.snake) > 1 and (new_direction[0] == -self.direction[0] and new_direction[1] == -self.direction[1]):
            new_direction = self.direction
        self.direction = new_direction

        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)
        
        # Default step reward.
        reward = -0.05

        # Check for wall collision.
        if not (0 <= new_head[0] < self.grid_width and 0 <= new_head[1] < self.grid_height):
            reward = -10
            self.done = True
            return self._get_observation(), reward, self.done, {}

        # Check for self-collision.
        if new_head in self.snake:
            reward = -10
            self.done = True
            return self._get_observation(), reward, self.done, {}

        # Move the snake by adding the new head.
        self.snake.insert(0, new_head)
        
        # Check if fruit is eaten.
        if new_head == self.fruit:
            reward = 10
            self.fruit = self._spawn_fruit()
        else:
            self.snake.pop()  # Remove the tail for a normal move.
        
        # Check for trap collision.
        if new_head in self.traps:
            new_length = (len(self.snake) + 1) // 2
            if new_length < 2:
                reward = -10
                self.done = True
                return self._get_observation(), reward, self.done, {}
            else:
                self.snake = self.snake[:new_length]
                reward = -5

        self.step_count += 1
        # Every 10 steps, spawn 2 new traps.
        if self.step_count % 10 == 0:
            for _ in range(2):
                trap = self._spawn_trap()
                self.traps.append(trap)
        
        # Optionally end the episode after 500 steps.
        if self.step_count > 500:
            self.done = True

        # Optional reward shaping: bonus for reducing Manhattan distance to the fruit.
        distance = abs(new_head[0] - self.fruit[0]) + abs(new_head[1] - self.fruit[1])
        reward += (self.previous_distance - distance) * 0.05
        self.previous_distance = distance

        return self._get_observation(), reward, self.done, {}

    def render(self, mode="human"):
        # Initialize Pygame display if not already done.
        if not hasattr(self, "screen"):
            pygame.init()
            self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
            pygame.display.set_caption("Snake Game")
            self.clock = pygame.time.Clock()
        
        # Process events (to allow window closing, etc.).
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit()
        
        # Fill the background.
        self.screen.fill(BLACK)
        
        # Draw the snake.
        for pos in self.snake:
            rect = pygame.Rect(pos[0] * CELL_SIZE, pos[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, GREEN, rect)
        
        # Draw the fruit.
        fx, fy = self.fruit
        fruit_rect = pygame.Rect(fx * CELL_SIZE, fy * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(self.screen, RED, fruit_rect)
        
        # Draw the traps.
        for trap in self.traps:
            trap_rect = pygame.Rect(trap[0] * CELL_SIZE, trap[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(self.screen, ORANGE, trap_rect)
        
        # Update the display.
        pygame.display.flip()
        self.clock.tick(10)  # Limit to 10 FPS

# ---------------------------
# Main Section: Training then Running Pygame
# ---------------------------
if __name__ == "__main__":
    env = SnakeEnv()
    # Train the agent using PPO.
    model = PPO("MlpPolicy", env, verbose=1)
    print("Starting training...")
    model.learn(total_timesteps=100000)  # Increase timesteps as needed.
    model.save("snake_ppo_model")
    print("Training finished and model saved.")

    # After training, run the evaluation with a Pygame window.
    obs = env.reset()
    done = False
    total_reward = 0
    print("Starting evaluation. Close the window to exit.")
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        env.render()
        time.sleep(0.1)  # Slow down the loop for viewing
    
    print("Episode finished with total reward:", total_reward)
    time.sleep(2)
    pygame.quit()
