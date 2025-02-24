import pygame
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ---------------------------
# Configurações do jogo
# ---------------------------
GRID_WIDTH = 30
GRID_HEIGHT = 20
BLOCK_SIZE = 20

WINDOW_WIDTH = GRID_WIDTH * BLOCK_SIZE
WINDOW_HEIGHT = GRID_HEIGHT * BLOCK_SIZE

# Cores
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED   = (255, 0, 0)
BLUE  = (0, 0, 255)

# ---------------------------
# Definição da Rede DQN
# ---------------------------
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

# ---------------------------
# Ambiente do Jogo da Cobrinha com Pygame
# ---------------------------
class SnakeGame:
    def __init__(self):
        self.grid_width = GRID_WIDTH
        self.grid_height = GRID_HEIGHT
        self.reset()
        # Inicializa o Pygame
        self.screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        pygame.display.set_caption("Snake RL - Modelo Carregado")
        self.clock = pygame.time.Clock()
        self.FPS = 10  # ajuste a velocidade se necessário
    
    def reset(self):
        # Inicia a cobra com 3 blocos no centro, orientada para a direita
        self.snake = [
            (self.grid_width // 2, self.grid_height // 2),
            (self.grid_width // 2 - 1, self.grid_height // 2),
            (self.grid_width // 2 - 2, self.grid_height // 2)
        ]
        self.direction = (1, 0)
        self.obstacles = []  # inicia sem obstáculos; serão adicionados com o tempo
        self.food = self.place_food()
        self.steps = 0
        self.obstacle_interval = 50  # a cada 50 passos, um novo obstáculo é adicionado
        self.done = False
        return self.get_state()
    
    def place_food(self):
        while True:
            pos = (random.randint(0, self.grid_width - 1),
                   random.randint(0, self.grid_height - 1))
            if pos not in self.snake and pos not in self.obstacles:
                return pos

    def add_obstacle(self):
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
        Atualiza o estado, retorna (state, reward, done).
        """
        # Atualiza a direção, impedindo reverter imediatamente
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
        reward = -0.1

        # Adiciona obstáculo periodicamente
        if self.steps % self.obstacle_interval == 0:
            self.add_obstacle()

        # Verifica colisão com a parede
        if not (0 <= new_head[0] < self.grid_width and 0 <= new_head[1] < self.grid_height):
            self.done = True
            reward = -10
            return self.get_state(), reward, self.done

        # Verifica colisão com o próprio corpo
        if new_head in self.snake:
            self.done = True
            reward = -10
            return self.get_state(), reward, self.done

        # Move a cobra
        self.snake.insert(0, new_head)
        if new_head == self.food:
            reward = 10
            self.food = self.place_food()
        else:
            self.snake.pop()

        # Se colidir com obstáculo, aplica penalidade (remove bloco extra)
        if new_head in self.obstacles:
            if len(self.snake) > 1:
                self.snake.pop()
                reward = -5
            else:
                self.done = True
                reward = -10

        return self.get_state(), reward, self.done

    def get_state(self):
        """
        Representa o estado como um array numpy com 3 canais:
          Canal 0: cobra, Canal 1: comida, Canal 2: obstáculos.
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
        self.screen.fill(BLACK)
        # Desenha a cobra
        for (x, y) in self.snake:
            rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.screen, GREEN, rect)
        # Desenha a comida
        fx, fy = self.food
        rect = pygame.Rect(fx * BLOCK_SIZE, fy * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
        pygame.draw.rect(self.screen, RED, rect)
        # Desenha os obstáculos
        for (x, y) in self.obstacles:
            rect = pygame.Rect(x * BLOCK_SIZE, y * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
            pygame.draw.rect(self.screen, BLUE, rect)
        pygame.display.flip()
    
    def handle_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
    
    def tick(self):
        self.clock.tick(self.FPS)

# ---------------------------
# Função principal: Carrega o modelo e joga
# ---------------------------
def main():
    pygame.init()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Inicializa o ambiente do jogo
    env = SnakeGame()
    state_shape = env.get_state().shape
    n_actions = 4

    # Cria a rede e carrega o modelo salvo
    model = DQN(state_shape, n_actions).to(device)
    model.load_state_dict(torch.load("snake_dqn.pth", map_location=device))
    model.eval()

    state = env.reset()
    done = False

    while True:
        env.handle_events()

        # Prepara o estado para o modelo: tensor de tamanho (1, canais, H, W)
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            q_values = model(state_tensor)
        action = q_values.max(1)[1].item()

        state, reward, done = env.step(action)
        env.render()
        env.tick()

        if done:
            pygame.time.wait(1000)
            state = env.reset()
            done = False

if __name__ == '__main__':
    main()
