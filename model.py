import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

# --------------------------
# Definição do ambiente Snake
# --------------------------

GRID_WIDTH = 30
GRID_HEIGHT = 20

class SnakeEnv:
    def __init__(self):
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_HEIGHT
        self.reset()

    def reset(self):
        # Define os obstáculos antes de chamar _place_food()
        self.obstacles = []  # Inicia sem obstáculos; serão adicionados com o tempo
        # Inicia a cobra com 3 blocos no centro (orientada para a direita)
        self.snake = [
            (self.grid_width // 2, self.grid_height // 2),
            (self.grid_width // 2 - 1, self.grid_height // 2),
            (self.grid_width // 2 - 2, self.grid_height // 2)
        ]
        self.direction = (1, 0)  # direção inicial: para a direita
        self.food = self._place_food()
        self.steps = 0
        self.obstacle_interval = 50  # a cada 50 passos, um novo obstáculo é adicionado
        self.done = False
        return self._get_state()

    def _place_food(self):
        while True:
            pos = (random.randint(0, self.grid_width - 1),
                   random.randint(0, self.grid_height - 1))
            if pos not in self.snake and pos not in self.obstacles:
                return pos


    def _add_obstacle(self):
        attempts = 0
        while attempts < 100:
            pos = (random.randint(0, self.grid_width - 1),
                   random.randint(0, self.grid_height - 1))
            if pos not in self.snake and pos not in self.obstacles and pos != self.food:
                self.obstacles.append(pos)
                break
            attempts += 1

    def step(self, action):
        """
        Ações: 0 = cima, 1 = baixo, 2 = esquerda, 3 = direita.
        Atualiza o estado, aplica recompensas e verifica condições de término.
        """
        # Atualiza a direção com base na ação (impede reversão imediata)
        if action == 0 and self.direction != (0, 1):
            self.direction = (0, -1)
        elif action == 1 and self.direction != (0, -1):
            self.direction = (0, 1)
        elif action == 2 and self.direction != (1, 0):
            self.direction = (-1, 0)
        elif action == 3 and self.direction != (-1, 0):
            self.direction = (1, 0)

        head = self.snake[0]
        new_head = (head[0] + self.direction[0], head[1] + self.direction[1])
        self.steps += 1
        reward = -0.1  # penalidade pequena a cada passo

        # Adiciona obstáculo periodicamente
        if self.steps % self.obstacle_interval == 0:
            self._add_obstacle()

        # Verifica colisão com parede
        if not (0 <= new_head[0] < self.grid_width and 0 <= new_head[1] < self.grid_height):
            self.done = True
            reward = -10
            return self._get_state(), reward, self.done, {}

        # Colisão com o próprio corpo
        if new_head in self.snake:
            self.done = True
            reward = -10
            return self._get_state(), reward, self.done, {}

        # Move a cobra
        self.snake.insert(0, new_head)
        # Se comer a comida, aumenta e gera nova comida; caso contrário, remove a cauda
        if new_head == self.food:
            reward = 10
            self.food = self._place_food()
        else:
            self.snake.pop()

        # Se colidir com obstáculo, aplica penalidade (reduz tamanho)
        if new_head in self.obstacles:
            if len(self.snake) > 1:
                self.snake.pop()  # remoção extra de um bloco
                reward = -5
            else:
                self.done = True
                reward = -10

        return self._get_state(), reward, self.done, {}

    def _get_state(self):
        """
        Representa o estado como um tensor 3xHxW:
         - Canal 0: posição da cobra (1 onde está a cobra)
         - Canal 1: posição da comida
         - Canal 2: posição dos obstáculos
        """
        state = np.zeros((3, self.grid_height, self.grid_width), dtype=np.float32)
        for (x, y) in self.snake:
            state[0, y, x] = 1.0
        fx, fy = self.food
        state[1, fy, fx] = 1.0
        for (x, y) in self.obstacles:
            state[2, y, x] = 1.0
        return state

    def render(self):
        # Renderização simples no terminal (opcional)
        grid = [[' ' for _ in range(self.grid_width)] for _ in range(self.grid_height)]
        for (x, y) in self.obstacles:
            grid[y][x] = 'X'
        for (x, y) in self.snake:
            grid[y][x] = 'O'
        fx, fy = self.food
        grid[fy][fx] = '*'
        print('\n'.join([''.join(row) for row in grid]))
        print('-' * self.grid_width)

# --------------------------
# Definição da rede DQN (Deep Q-Network)
# --------------------------

class DQN(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DQN, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1),
            nn.ReLU()
        )
        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        conv_out = self.conv(x)
        conv_out = conv_out.view(x.size()[0], -1)
        return self.fc(conv_out)

# --------------------------
# Replay Buffer para amostragem de experiências
# --------------------------

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = map(np.array, zip(*batch))
        return state, action, reward, next_state, done
    
    def __len__(self):
        return len(self.buffer)

# --------------------------
# Hiperparâmetros e inicialização do treinamento
# --------------------------

EPISODES = 30          # Número de episódios de treinamento
BATCH_SIZE = 64
GAMMA = 0.99
LEARNING_RATE = 1e-3
REPLAY_BUFFER_CAPACITY = 10000
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 300         # Taxa de decaimento do epsilon para a política ε-greedy

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = SnakeEnv()
n_actions = 4
state_shape = env._get_state().shape

policy_net = DQN(state_shape, n_actions).to(device)
target_net = DQN(state_shape, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
replay_buffer = ReplayBuffer(REPLAY_BUFFER_CAPACITY)

def select_action(state, steps_done):
    # Estratégia ε-greedy: com probabilidade eps escolhe ação aleatória, senão escolhe a melhor ação segundo o modelo
    eps_threshold = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    if random.random() < eps_threshold:
        return random.randrange(n_actions)
    else:
        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            q_values = policy_net(state_tensor)
            return q_values.max(1)[1].item()

def compute_loss(batch):
    states, actions, rewards, next_states, dones = batch
    states = torch.tensor(states, dtype=torch.float32).to(device)
    actions = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(device)
    rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
    next_states = torch.tensor(next_states, dtype=torch.float32).to(device)
    dones = torch.tensor(dones, dtype=torch.float32).to(device)

    q_values = policy_net(states).gather(1, actions).squeeze(1)
    next_q_values = target_net(next_states).max(1)[0]
    expected_q_values = rewards + GAMMA * next_q_values * (1 - dones)
    loss = F.mse_loss(q_values, expected_q_values.detach())
    return loss

steps_done = 0
update_target_every = 1000

# --------------------------
# Loop de treinamento
# --------------------------

for episode in range(EPISODES):
    state = env.reset()
    total_reward = 0
    while True:
        action = select_action(state, steps_done)
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        replay_buffer.push(state, action, reward, next_state, done)
        state = next_state
        steps_done += 1

        if len(replay_buffer) >= BATCH_SIZE:
            batch = replay_buffer.sample(BATCH_SIZE)
            loss = compute_loss(batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        if steps_done % update_target_every == 0:
            target_net.load_state_dict(policy_net.state_dict())
        
        if done:
            break
    print(f"Episode {episode+1}, Recompensa Total: {total_reward}")

# --------------------------
# Salvando o modelo treinado
# --------------------------

torch.save(policy_net.state_dict(), "snake_dqn.pth")
print("Modelo salvo em 'snake_dqn.pth'")
