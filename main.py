import pygame
import random
from collections import deque

# ---------------------------
# Game Configuration
# ---------------------------
CELL_SIZE = 20         # Pixel size of each grid cell.
GRID_WIDTH = 20        # Number of cells horizontally.
GRID_HEIGHT = 20       # Number of cells vertically.
SCREEN_WIDTH = CELL_SIZE * GRID_WIDTH
SCREEN_HEIGHT = CELL_SIZE * GRID_HEIGHT

# Colors (RGB)
BLACK   = (0, 0, 0)         # Background
GREEN   = (0, 255, 0)       # Snake color
RED     = (255, 0, 0)       # Fruit color
WHITE   = (255, 255, 255)   # Text color
TRAP_COLOR = (255, 165, 0)  # Orange – trap color

# Trap settings
TRAP_CUT_LENGTH = 3       # How many segments to cut off when a trap is hit.

# ---------------------------
# Helper Functions
# ---------------------------
def spawn_fruit(snake, traps):
    """Return a random position (tuple) that is not occupied by the snake or a trap."""
    while True:
        pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if pos not in snake and pos not in traps:
            return pos

def spawn_trap(traps, snake, fruit):
    """Return a random position (tuple) that is not occupied by the snake, fruit, or existing traps."""
    while True:
        pos = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        if pos not in snake and pos != fruit and pos not in traps:
            return pos

def find_path(snake, fruit, traps):
    """
    Use BFS to try to find a sequence of moves that leads the snake from its current state
    to the fruit while avoiding traps.
    
    The BFS state is a tuple: (snake_state, path) where:
      - snake_state is a tuple of positions (with head at index 0)
      - path is a list of moves (each move is a tuple, e.g. (1,0) for right)
      
    Movement simulation:
      - Without eating fruit: new snake state = (new_head,) + snake_state[:-1]
      - When fruit is reached: the snake grows (tail is not removed).
      
    We “free” the tail cell when moving normally (since it will be removed) and skip any move
    that would land on a trap.
    """
    directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # Up, Down, Left, Right
    initial_state = (tuple(snake), [])
    visited = set([tuple(snake)])
    queue = deque([initial_state])
    
    while queue:
        snake_state, path = queue.popleft()
        head = snake_state[0]
        
        for d in directions:
            new_head = (head[0] + d[0], head[1] + d[1])
            # Check boundaries.
            if not (0 <= new_head[0] < GRID_WIDTH and 0 <= new_head[1] < GRID_HEIGHT):
                continue

            # Skip if new_head is on a trap.
            if new_head in traps:
                continue

            # For a normal move the tail will move (freeing up that cell).
            occupied = set(snake_state)
            if len(snake_state) > 1:
                occupied.remove(snake_state[-1])
            
            if new_head in occupied:
                continue

            if new_head == fruit:
                # Eat the fruit: snake grows (tail is not removed)
                new_snake = (new_head,) + snake_state
                new_path = path + [d]
                return new_path
            else:
                # Normal move: add new head and remove tail.
                new_snake = (new_head,) + snake_state[:-1]
            
            if new_snake in visited:
                continue
            visited.add(new_snake)
            queue.append((new_snake, path + [d]))
    
    # No path found.
    return None

def get_next_direction(snake, fruit, current_direction, traps):
    """
    Decide the next move for the snake.
    First, try to find a complete path to the fruit (avoiding traps).
    If one is found, return its first move.
    Otherwise, choose any safe move that avoids immediate collisions with walls, itself, or traps.
    """
    path = find_path(snake, fruit, traps)
    if path is not None and len(path) > 0:
        return path[0]
    else:
        # Fallback: try all four directions.
        directions = [(0, -1), (0, 1), (-1, 0), (1, 0)]
        head = snake[0]
        for d in directions:
            new_head = (head[0] + d[0], head[1] + d[1])
            if not (0 <= new_head[0] < GRID_WIDTH and 0 <= new_head[1] < GRID_HEIGHT):
                continue
            if new_head in snake[:-1]:
                continue
            if new_head in traps:
                continue
            return d
        # If no safe move is found, continue in the current direction.
        return current_direction

# ---------------------------
# Main Game Loop
# ---------------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Snake AI with Traps")
    clock = pygame.time.Clock()
    
    # Set a timer event for trap spawning every 2000 ms.
    TRAP_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(TRAP_EVENT, 2000)
    
    # Initialize font for displaying score.
    font = pygame.font.SysFont(None, 36)
    
    # Initialize the snake: list of positions (head at index 0).
    snake = [(GRID_WIDTH // 2, GRID_HEIGHT // 2),
             (GRID_WIDTH // 2 - 1, GRID_HEIGHT // 2),
             (GRID_WIDTH // 2 - 2, GRID_HEIGHT // 2)]
    current_direction = (1, 0)  # Start moving to the right.
    
    # Initialize fruit and traps.
    traps = []
    fruit = spawn_fruit(snake, traps)
    
    score = 0
    running = True

    while running:
        # Process events (quit, trap spawning, etc.).
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == TRAP_EVENT:
                # Spawn a new trap every 2 seconds.
                trap_pos = spawn_trap(traps, snake, fruit)
                traps.append(trap_pos)
        
        # --- AI Decides the Next Move ---
        next_direction = get_next_direction(snake, fruit, current_direction, traps)
        current_direction = next_direction
        
        # Compute the new head position.
        head = snake[0]
        new_head = (head[0] + current_direction[0], head[1] + current_direction[1])
        
        # Check for wall collisions.
        if not (0 <= new_head[0] < GRID_WIDTH and 0 <= new_head[1] < GRID_HEIGHT):
            print("Game Over! Hit a wall. Final score:", score)
            running = False
            continue
        
        # Check for self–collision.
        # (Note: when not eating a fruit, the snake’s tail moves, so the last cell is safe.)
        if new_head in snake[:-1]:
            print("Game Over! Ran into itself. Final score:", score)
            running = False
            continue
        
        # Move the snake.
        if new_head == fruit:
            # Eat the fruit: grow the snake and increase score.
            snake = [new_head] + snake
            score += 1
            fruit = spawn_fruit(snake, traps)
        else:
            # Normal move: add new head and remove tail.
            snake = [new_head] + snake[:-1]
        
        # --- Trap Collision Check ---
        # If the snake’s head lands on a trap, cut TRAP_CUT_LENGTH segments off its tail.
        if snake[0] in traps:
            if len(snake) <= TRAP_CUT_LENGTH:
                print("Game Over! Hit a trap and lost too much of the snake. Final score:", score)
                running = False
                continue
            else:
                snake = snake[:-TRAP_CUT_LENGTH]
                print("Hit a trap! Snake cut by", TRAP_CUT_LENGTH, "segments. New length:", len(snake))
        
        # --- Drawing ---
        screen.fill(BLACK)
        
        # Draw the fruit.
        fruit_rect = pygame.Rect(fruit[0] * CELL_SIZE, fruit[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(screen, RED, fruit_rect)
        
        # Draw traps.
        for trap in traps:
            trap_rect = pygame.Rect(trap[0] * CELL_SIZE, trap[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, TRAP_COLOR, trap_rect)
        
        # Draw the snake.
        for segment in snake:
            segment_rect = pygame.Rect(segment[0] * CELL_SIZE, segment[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            pygame.draw.rect(screen, GREEN, segment_rect)
        
        # Draw the score on the screen.
        score_text = font.render("Score: " + str(score), True, WHITE)
        screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        clock.tick(10)  # Adjust speed (frames per second) as desired.
    
    pygame.quit()

if __name__ == '__main__':
    main()
