import gym
from gym import spaces
import numpy as np
import random

# ---------------------------
# 1. Define the Snake Environment
# ---------------------------
class SnakeEnv(gym.Env):
    """
    A simple Snake environment with traps.

    Grid values:
      0: empty cell
      1: snake body
      2: fruit
      3: trap
    """
    metadata = {"render.modes": ["human"]}
    
    def __init__(self):
        super(SnakeEnv, self).__init__()
        self.grid_width = 20
        self.grid_height = 20
        
        # Observation: flattened grid (values 0-3). Shape: (grid_width*grid_height,)
        self.observation_space = spaces.Box(low=0, high=3,
                                            shape=(self.grid_width * self.grid_height,),
                                            dtype=np.uint8)
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right.
        self.action_space = spaces.Discrete(4)
        self.reset()

    def reset(self):
        self.snake = [(self.grid_width // 2, self.grid_height // 2)]
        self.direction = (1, 0)  # start moving right
        self.traps = []         # <-- Initialize traps BEFORE spawning fruit
        self.fruit = self._spawn_fruit()
        self.step_count = 0
        self.done = False
        return self._get_observation()

    def _spawn_fruit(self):
        while True:
            pos = (random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1))
            if pos not in self.snake and pos not in self.traps:
                return pos

    def _spawn_trap(self):
        while True:
            pos = (random.randint(0, self.grid_width - 1), random.randint(0, self.grid_height - 1))
            if pos not in self.snake and pos != self.fruit and pos not in self.traps:
                return pos

    def _get_observation(self):
        grid = np.zeros((self.grid_width, self.grid_height), dtype=np.uint8)
        # Mark snake positions
        for pos in self.snake:
            grid[pos[0], pos[1]] = 1
        # Mark fruit
        fx, fy = self.fruit
        grid[fx, fy] = 2
        # Mark traps
        for trap in self.traps:
            grid[trap[0], trap[1]] = 3
        return grid.flatten()

    def step(self, action):
        if self.done:
            return self._get_observation(), 0, True, {}
        
        # Map action to direction
        if action == 0:
            new_direction = (0, -1)
        elif action == 1:
            new_direction = (0, 1)
        elif action == 2:
            new_direction = (-1, 0)
        elif action == 3:
            new_direction = (1, 0)
        else:
            new_direction = self.direction
        
        # Prevent reversal (if snake length > 1)
        if len(self.snake) > 1:
            if (new_direction[0] == -self.direction[0] and new_direction[1] == -self.direction[1]):
                new_direction = self.direction
        self.direction = new_direction

        head_x, head_y = self.snake[0]
        dx, dy = self.direction
        new_head = (head_x + dx, head_y + dy)
        reward = -0.1  # small step penalty

        # Check wall collision
        if not (0 <= new_head[0] < self.grid_width and 0 <= new_head[1] < self.grid_height):
            reward = -10
            self.done = True
            return self._get_observation(), reward, self.done, {}
        
        # Check self-collision
        if new_head in self.snake:
            reward = -10
            self.done = True
            return self._get_observation(), reward, self.done, {}
        
        # Move snake: add new head
        self.snake.insert(0, new_head)
        
        # Check if fruit eaten
        if new_head == self.fruit:
            reward = 10
            self.fruit = self._spawn_fruit()
        else:
            # Normal move: remove tail
            self.snake.pop()
        
        # Check trap collision: if the snake hits a trap, cut its length roughly in half.
        if new_head in self.traps:
            new_length = (len(self.snake) + 1) // 2
            if new_length < 2:
                reward = -10
                self.done = True
                return self._get_observation(), reward, self.done, {}
            else:
                self.snake = self.snake[:new_length]
                reward = -5  # penalty for hitting a trap

        self.step_count += 1
        # Every 10 steps (~1 second if 10 steps per second), spawn 2 new traps.
        if self.step_count % 10 == 0:
            for _ in range(2):
                trap = self._spawn_trap()
                self.traps.append(trap)
        
        # Optionally end an episode after a maximum number of steps.
        if self.step_count > 500:
            self.done = True

        return self._get_observation(), reward, self.done, {}

    def render(self, mode="human"):
        # A very simple text-based render.
        grid = np.full((self.grid_width, self.grid_height), '.', dtype=str)
        for pos in self.snake:
            grid[pos[0], pos[1]] = 'S'
        fx, fy = self.fruit
        grid[fx, fy] = 'F'
        for trap in self.traps:
            grid[trap[0], trap[1]] = 'T'
        for y in range(self.grid_height):
            print(' '.join(grid[:, y]))
        print()

# ---------------------------
# 2. Train a DQN Agent using Stable Baselines3
# ---------------------------
if __name__ == "__main__":
    # Uncomment the following block to do a quick manual test of the environment:
    """
    env = SnakeEnv()
    obs = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # random action for testing
        obs, reward, done, info = env.step(action)
        env.render()
        print("Reward:", reward)
    """
    
    # Training with DQN (the simplest deep RL approach):
    from stable_baselines3 import DQN

    env = SnakeEnv()
    model = DQN("MlpPolicy", env, verbose=1)
    # Train for 10,000 timesteps (adjust as needed)
    model.learn(total_timesteps=10000)
    model.save("snake_dqn_model")

    # To see the trained agent in action:
    obs = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        env.render()
