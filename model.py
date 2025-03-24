import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

# Hiperparâmetros
MAX_MEMORY = 100000
BATCH_SIZE = 1000
LR = 0.001
GAMMA = 0.9
EPSILON_START = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995

# Parâmetros do ambiente
GRID_WIDTH = 30
GRID_HEIGHT = 20
ADD_ITEM_EVERY = 20  # A cada 20 passos (aprox. 2 seg se FPS=10), adiciona recompensa e obstáculos

# Definindo as direções (Up, Right, Down, Left)
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]


class SnakeGameAI:
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
        self.direction = (0, -1)  # inicia subindo
        self.score = 0
        self.rewards_list = []
        self.obstacles = []
        self.frame_iteration = 0
        return self.get_state()
    
    def add_items(self):
        occupied = set(self.snake) | set(self.rewards_list) | set(self.obstacles)
        # Adiciona uma recompensa
        pos = self.random_position(occupied)
        self.rewards_list.append(pos)
        # Adiciona dois obstáculos
        for _ in range(2):
            occupied = set(self.snake) | set(self.rewards_list) | set(self.obstacles)
            pos = self.random_position(occupied)
            self.obstacles.append(pos)
    
    def random_position(self, occupied):
        while True:
            pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if pos not in occupied:
                return pos
    
    def is_collision(self, point=None):
        if point is None:
            point = self.snake[0]
        # Colisão com borda
        if point[0] < 0 or point[0] >= GRID_WIDTH or point[1] < 0 or point[1] >= GRID_HEIGHT:
            return True
        # Colisão com si mesmo
        if point in self.snake[1:]:
            return True
        return False

    def move(self, action):
        """
        Ação: 0 - seguir em frente, 1 - virar à direita, 2 - virar à esquerda.
        A direção é atualizada de forma relativa.
        """
        idx = DIRECTIONS.index(self.direction)
        if action == 1:  # virar à direita
            new_idx = (idx + 1) % 4
        elif action == 2:  # virar à esquerda
            new_idx = (idx - 1) % 4
        else:
            new_idx = idx
        self.direction = DIRECTIONS[new_idx]
        new_head = (self.snake[0][0] + self.direction[0], self.snake[0][1] + self.direction[1])
        self.snake.insert(0, new_head)
    
    def update(self, action):
        self.frame_iteration += 1
        # A cada ADD_ITEM_EVERY passos, adiciona uma recompensa e dois obstáculos
        if self.frame_iteration % ADD_ITEM_EVERY == 0:
            self.add_items()
        
        self.move(action)
        reward = 0
        game_over = False
        head = self.snake[0]
        
        # Checa colisão com parede ou com o próprio corpo
        if self.is_collision(head):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        # Verifica se a cabeça está numa recompensa
        if head in self.rewards_list:
            self.rewards_list.remove(head)
            self.score += 1
            reward = 10
            # Cresce: não remove a cauda nesta jogada
        else:
            # Movimento normal: remove a cauda
            self.snake.pop()
        
        # Verifica colisão com obstáculos e remove 50% do tamanho
        if head in self.obstacles:
            self.obstacles.remove(head)
            if len(self.snake) > 1:
                # Calcula 50% do tamanho atual (garante pelo menos 1 bloco)
                new_length = max(1, len(self.snake) // 2)
                self.snake = self.snake[:new_length]
            else:
                game_over = True
                reward = -10
        
        return reward, game_over, self.score

    def get_state(self):
        head = self.snake[0]

        # Função auxiliar para verificar "perigo" em uma dada direção
        def danger_in_direction(direction):
            next_point = (head[0] + direction[0], head[1] + direction[1])
            if next_point[0] < 0 or next_point[0] >= GRID_WIDTH or next_point[1] < 0 or next_point[1] >= GRID_HEIGHT:
                return 1
            if next_point in self.snake[1:]:
                return 1
            if next_point in self.obstacles:
                return 1
            return 0
        
        # Calcula os perigos para frente, à direita e à esquerda, de acordo com a direção atual
        idx = DIRECTIONS.index(self.direction)
        straight = self.direction
        right = DIRECTIONS[(idx + 1) % 4]
        left = DIRECTIONS[(idx - 1) % 4]
        
        danger_straight = danger_in_direction(straight)
        danger_right = danger_in_direction(right)
        danger_left = danger_in_direction(left)
        
        # Codificação one-hot da direção atual (up, right, down, left)
        dir_up = 1 if self.direction == (0, -1) else 0
        dir_right = 1 if self.direction == (1, 0) else 0
        dir_down = 1 if self.direction == (0, 1) else 0
        dir_left = 1 if self.direction == (-1, 0) else 0
        
        # Posição relativa da recompensa (se houver)
        food_left = food_right = food_up = food_down = 0
        if self.rewards_list:
            closest = min(self.rewards_list, key=lambda p: abs(p[0] - head[0]) + abs(p[1] - head[1]))
            if closest[0] < head[0]:
                food_left = 1
            elif closest[0] > head[0]:
                food_right = 1
            if closest[1] < head[1]:
                food_up = 1
            elif closest[1] > head[1]:
                food_down = 1
        
        state = [
            danger_straight, danger_right, danger_left,
            dir_up, dir_right, dir_down, dir_left,
            food_left, food_right, food_up, food_down
        ]
        return np.array(state, dtype=int)


# Modelo de Rede Neural (DQN simples)
class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x


# Memória para Experience Replay
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)
    
    def push(self, transition):
        self.memory.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
    
    def __len__(self):
        return len(self.memory)
    
    # Permite iterar sobre os itens da memória
    def __iter__(self):
        return iter(self.memory)


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = EPSILON_START  # taxa de exploração
        self.gamma = GAMMA
        self.memory = ReplayMemory(MAX_MEMORY)
        self.model = Linear_QNet(11, 128, 3)
        self.optimizer = optim.Adam(self.model.parameters(), lr=LR)
        self.criterion = nn.MSELoss()
    
    def get_action(self, state):
        state0 = torch.tensor(state, dtype=torch.float)
        # Estratégia epsilon-greedy
        if random.random() < self.epsilon:
            move = random.randint(0, 2)
        else:
            with torch.no_grad():
                prediction = self.model(state0)
                move = torch.argmax(prediction).item()
        return move
    
    def train_short_memory(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float).unsqueeze(0)
        next_state = torch.tensor(next_state, dtype=torch.float).unsqueeze(0)
        reward = torch.tensor([reward], dtype=torch.float)
        action = torch.tensor([action], dtype=torch.long)
        done = torch.tensor([done], dtype=torch.bool)
        
        pred = self.model(state)
        pred = pred.gather(1, action.unsqueeze(1)).squeeze(1)
        target = reward + self.gamma * torch.max(self.model(next_state), dim=1)[0] * (not done)
        loss = self.criterion(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def train_long_memory(self):
        if len(self.memory) < BATCH_SIZE:
            mini_sample = list(self.memory)
        else:
            mini_sample = self.memory.sample(BATCH_SIZE)
        
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        states = torch.tensor(np.array(states), dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float)
        dones = torch.tensor(dones, dtype=torch.bool)
        
        pred = self.model(states)
        pred = pred.gather(1, actions.unsqueeze(1)).squeeze(1)
        target = rewards + self.gamma * torch.max(self.model(next_states), dim=1)[0] * (~dones)
        loss = self.criterion(pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def train():
    agent = Agent()
    game = SnakeGameAI()
    episodes = 500
    scores = []
    total_rewards = []
    
    for e in range(episodes):
        state = game.reset()
        done = False
        episode_reward = 0
        while not done:
            action = agent.get_action(state)
            reward, done, score = game.update(action)
            episode_reward += reward
            next_state = game.get_state()
            agent.train_short_memory(state, action, reward, next_state, done)
            agent.memory.push((state, action, reward, next_state, done))
            state = next_state
        agent.train_long_memory()
        scores.append(score)
        total_rewards.append(episode_reward)
        agent.epsilon = max(EPSILON_MIN, agent.epsilon * EPSILON_DECAY)
        print(f"Episode {e+1}: Score: {score}, Total Reward: {episode_reward}")
    
    # Salva o modelo treinado
    torch.save(agent.model.state_dict(), "model.pth")
    # Salva os resultados do treinamento
    with open("training_results.txt", "w") as f:
        for e, (s, r) in enumerate(zip(scores, total_rewards), start=1):
            f.write(f"Episode {e}: Score: {s}, Total Reward: {r}\n")

if __name__ == '__main__':
    train()
