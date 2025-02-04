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
BLACK      = (0, 0, 0)         # Background
GREEN      = (0, 255, 0)       # Snake color
RED        = (255, 0, 0)       # Fruit color
WHITE      = (255, 255, 255)   # Text color
TRAP_COLOR = (255, 165, 0)     # Orange – trap color

# ---------------------------
# Helper Functions
# ---------------------------
def spawn_fruit(snake, traps):
    """Return a random position (tuple) that is not occupied by the snake or any trap."""
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
    Use Breadth-First Search (BFS) to find a sequence of moves that leads the snake from its current state
    to the fruit while avoiding traps and self-collision.
    
    The BFS state is a tuple: (snake_state, path) where:
      - snake_state is a tuple of positions (with head at index 0)
      - path is a list of moves (each move is a tuple, e.g., (1,0) for right)
      
    For a normal move (when not eating fruit) the tail is freed so that cell is not considered blocked.
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
            # Check board boundaries.
            if not (0 <= new_head[0] < GRID_WIDTH and 0 <= new_head[1] < GRID_HEIGHT):
                continue

            # Avoid trap cells.
            if new_head in traps:
                continue

            # For a normal move the tail will move, so remove the tail cell from occupied.
            occupied = set(snake_state)
            if len(snake_state) > 1:
                occupied.remove(snake_state[-1])
            
            if new_head in occupied:
                continue

            if new_head == fruit:
                # Reaching the fruit: snake grows (tail not removed).
                new_snake = (new_head,) + snake_state
                return path + [d]
            else:
                # Normal move: add new head and remove tail.
                new_snake = (new_head,) + snake_state[:-1]
            
            if new_snake in visited:
                continue
            
            visited.add(new_snake)
            queue.append((new_snake, path + [d]))
    
    # No valid path found.
    return None

def get_next_direction(snake, fruit, current_direction, traps):
    """
    Decide the next move for the snake.
    First try to find a complete path to the fruit (avoiding traps).
    If found, return the first move of that path.
    Otherwise, pick any safe move (avoiding walls, self, and traps).
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
    pygame.display.set_caption("Snake AI with 2 Traps per Second")
    clock = pygame.time.Clock()
    
    # Set a timer event to spawn traps every 1000 ms.
    TRAP_EVENT = pygame.USEREVENT + 1
    pygame.time.set_timer(TRAP_EVENT, 1000)
    
    # Initialize font for displaying the score.
    font = pygame.font.SysFont(None, 36)
    
    # Initialize the snake: head is at index 0.
    snake = [
        (GRID_WIDTH // 2, GRID_HEIGHT // 2),
        (GRID_WIDTH // 2 - 1, GRID_HEIGHT // 2),
        (GRID_WIDTH // 2 - 2, GRID_HEIGHT // 2)
    ]
    current_direction = (1, 0)  # Starting to move to the right.
    
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
                # Spawn 2 traps every second.
                for _ in range(2):
                    trap_pos = spawn_trap(traps, snake, fruit)
                    traps.append(trap_pos)
        
        # --- AI Decides the Next Move ---
        next_direction = get_next_direction(snake, fruit, current_direction, traps)
        current_direction = next_direction

        # Compute new head position.
        head = snake[0]
        new_head = (head[0] + current_direction[0], head[1] + current_direction[1])
        
        # Check for collision with walls.
        if not (0 <= new_head[0] < GRID_WIDTH and 0 <= new_head[1] < GRID_HEIGHT):
            print("Game Over! Hit a wall. Final score:", score)
            running = False
            continue
        
        # Check for self–collision.
        if new_head in snake[:-1]:
            print("Game Over! Ran into itself. Final score:", score)
            running = False
            continue
        
        # Move the snake.
        if new_head == fruit:
            # Eat the fruit: snake grows and score increases.
            snake = [new_head] + snake
            score += 1
            fruit = spawn_fruit(snake, traps)
        else:
            # Normal move: add new head and remove the tail.
            snake = [new_head] + snake[:-1]
        
        # --- Trap Collision Check ---
        if snake[0] in traps:
            # When hitting a trap, cut the snake in half.
            # We use rounding up so that, for example, a 3–segment snake becomes 2 segments.
            new_length = (len(snake) + 1) // 2
            if new_length < 2:
                print("Game Over! Hit a trap and lost half its length. Final score:", score)
                running = False
                continue
            else:
                snake = snake[:new_length]
                print("Hit a trap! Snake is cut to half its length. New length:", len(snake))
        
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
        
        # Draw the score.
        score_text = font.render("Score: " + str(score), True, WHITE)
        screen.blit(score_text, (10, 10))
        
        pygame.display.flip()
        clock.tick(10)  # Adjust frames per second as desired.
    
    pygame.quit()

if __name__ == '__main__':
    main()
