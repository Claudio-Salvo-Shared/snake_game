import pygame
import random
import sys
import torch
import numpy as np

# Parâmetros da tela e grade
BLOCK_SIZE = 20
GRID_WIDTH = 30
GRID_HEIGHT = 20
WIDTH = GRID_WIDTH * BLOCK_SIZE
HEIGHT = GRID_HEIGHT * BLOCK_SIZE

# Cores
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
BLUE  = (0, 0, 255)
RED   = (255, 0, 0)
WHITE = (255, 255, 255)

# Definindo as direções (Up, Right, Down, Left)
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Jogo da Cobrinha Autônomo com Modelo Treinado")
clock = pygame.time.Clock()
FPS = 10

# Eventos de temporização
REWARD_EVENT = pygame.USEREVENT + 1
OBSTACLE_EVENT = pygame.USEREVENT + 2
pygame.time.set_timer(REWARD_EVENT, 2000)
pygame.time.set_timer(OBSTACLE_EVENT, 2000)

# Função para desenhar blocos
def draw_block(color, pos):
    rect = pygame.Rect(pos[0] * BLOCK_SIZE, pos[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE)
    pygame.draw.rect(screen, color, rect)

# Retorna uma posição aleatória não ocupada
def random_position(occupied):
    while True:
        pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if pos not in occupied:
            return pos

# Reinicia o jogo
def reset_game():
    snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2)]
    # Inicia com direção para cima
    direction = (0, -1)
    score = 0
    rewards = []
    obstacles = []
    return snake, direction, score, rewards, obstacles

# Atualiza a direção a partir da ação relativa
# Ação: 0 - seguir em frente, 1 - virar à direita, 2 - virar à esquerda
def update_direction(current_direction, action):
    idx = DIRECTIONS.index(current_direction)
    if action == 1:
        new_idx = (idx + 1) % 4
    elif action == 2:
        new_idx = (idx - 1) % 4
    else:
        new_idx = idx
    return DIRECTIONS[new_idx]

# Extrai o estado atual (11 features) – semelhante ao usado no treinamento
def get_state(snake, direction, rewards, obstacles):
    head = snake[0]
    # Perigo: verifica se na direção especificada há colisão
    def danger(dir_vec):
        next_point = (head[0] + dir_vec[0], head[1] + dir_vec[1])
        if next_point[0] < 0 or next_point[0] >= GRID_WIDTH or next_point[1] < 0 or next_point[1] >= GRID_HEIGHT:
            return 1
        if next_point in snake:
            return 1
        if next_point in obstacles:
            return 1
        return 0
    
    idx = DIRECTIONS.index(direction)
    straight = direction
    right = DIRECTIONS[(idx + 1) % 4]
    left = DIRECTIONS[(idx - 1) % 4]
    
    danger_straight = danger(straight)
    danger_right = danger(right)
    danger_left = danger(left)
    
    # Codificação one-hot da direção
    dir_up    = 1 if direction == (0, -1) else 0
    dir_right = 1 if direction == (1, 0)  else 0
    dir_down  = 1 if direction == (0, 1)  else 0
    dir_left  = 1 if direction == (-1, 0) else 0
    
    # Recompensa: define a direção da recompensa mais próxima (se houver)
    food_left = food_right = food_up = food_down = 0
    if rewards:
        closest = min(rewards, key=lambda p: abs(p[0] - head[0]) + abs(p[1] - head[1]))
        if closest[0] < head[0]:
            food_left = 1
        elif closest[0] > head[0]:
            food_right = 1
        if closest[1] < head[1]:
            food_up = 1
        elif closest[1] > head[1]:
            food_down = 1
    state = np.array([danger_straight, danger_right, danger_left,
                      dir_up, dir_right, dir_down, dir_left,
                      food_left, food_right, food_up, food_down], dtype=int)
    return state

# Define a rede neural idêntica à usada no treinamento
class Linear_QNet(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Linear_QNet, self).__init__()
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = torch.relu(self.linear1(x))
        x = self.linear2(x)
        return x

# Carrega o modelo treinado
model = Linear_QNet(11, 128, 3)
model.load_state_dict(torch.load("model.pth"))
model.eval()

def main():
    snake, direction, score, rewards, obstacles = reset_game()
    running = True
    while running:
        clock.tick(FPS)
        # Processa eventos do Pygame
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            elif event.type == REWARD_EVENT:
                occupied = set(snake) | set(rewards) | set(obstacles)
                pos = random_position(occupied)
                rewards.append(pos)
            elif event.type == OBSTACLE_EVENT:
                for _ in range(2):
                    occupied = set(snake) | set(rewards) | set(obstacles)
                    pos = random_position(occupied)
                    obstacles.append(pos)
        
        # Obtém o estado atual e utiliza o modelo para escolher a ação
        state = get_state(snake, direction, rewards, obstacles)
        state_tensor = torch.tensor(state, dtype=torch.float)
        with torch.no_grad():
            prediction = model(state_tensor)
        action = torch.argmax(prediction).item()  # 0: frente, 1: direita, 2: esquerda
        
        # Atualiza a direção com base na ação escolhida
        direction = update_direction(direction, action)
        new_head = (snake[0][0] + direction[0], snake[0][1] + direction[1])
        new_snake = [new_head] + snake
        
        # Verifica colisão com bordas
        if not (0 <= new_head[0] < GRID_WIDTH and 0 <= new_head[1] < GRID_HEIGHT):
            snake, direction, score, rewards, obstacles = reset_game()
            continue
        
        # Verifica colisão com si mesmo
        if new_head in new_snake[1:]:
            snake, direction, score, rewards, obstacles = reset_game()
            continue
        
        # Verifica colisão com recompensas
        if new_head in rewards:
            rewards.remove(new_head)
            score += 1
            # Cresce: não remove a cauda
        else:
            new_snake.pop()  # movimento normal: remove a cauda
        
        # Verifica colisão com obstáculos e remove 50% do tamanho
        if new_head in obstacles:
            obstacles.remove(new_head)
            if len(new_snake) > 1:
                new_length = max(1, len(new_snake) // 2)
                new_snake = new_snake[:new_length]
            else:
                snake, direction, score, rewards, obstacles = reset_game()
                continue
        
        snake = new_snake
        
        # Atualiza a tela
        screen.fill(BLACK)
        for r in rewards:
            draw_block(BLUE, r)
        for o in obstacles:
            draw_block(RED, o)
        for block in snake:
            draw_block(GREEN, block)
        
        # Exibe a pontuação
        font = pygame.font.SysFont("arial", 20)
        text = font.render("Pontuação: " + str(score), True, WHITE)
        screen.blit(text, (5, 5))
        
        pygame.display.flip()

if __name__ == "__main__":
    main()
