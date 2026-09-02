import pygame 
import numpy as np
import random

BLOCK_SIZE = 20
GRID_WIDTH = 20
GRID_HEIGHT = 20
WINDOW_WIDTH = GRID_WIDTH*BLOCK_SIZE
WINDOW_HEIGHT = GRID_HEIGHT*BLOCK_SIZE
FPS = 10

BACKGROUND = (18, 20, 24)
BODY = (60, 200, 110)
HEAD = (150, 240, 170)
FOOD = (232, 84, 84)

class SnakeGame:
    def __init__(self, render=True, fps=FPS):
        # Training does not need a window, and drawing one caps it at the frame
        # rate. Headless runs roughly two orders of magnitude faster.
        self.render = render
        self.fps = fps
        pygame.init()
        if render:
            self.display = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
            pygame.display.set_caption("AI Snake")
            self.clock = pygame.time.Clock()
        else:
            self.display = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
            self.clock = None
        self.reset()
    def reset(self):
        self.direction = 'RIGHT'
        self.head = [GRID_WIDTH//2, GRID_HEIGHT//2]
        # The body sits behind the head on the same row: x decreases, y is the
        # head's y. This read self.head[0] for both and only worked because the
        # grid is square and the head starts at its centre.
        self.snake = [
            self.head[:],
            [self.head[0] - 1, self.head[1]],
            [self.head[0] - 2, self.head[1]],
        ]
        self.spawn_food()
        self.score = 0
        self.frame_iteration = 0
    def spawn_food(self):
        while True:
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            self.food = [x, y]
            if self.food not in self.snake:
                break
    def move(self, action):
        directions = ['UP', 'RIGHT', 'DOWN', 'LEFT']
        idx = directions.index(self.direction)

        if np.array_equal(action, [1,0,0]):
            new_dir = directions[idx]
        elif np.array_equal(action, [0,1,0]):
            new_dir = directions[(idx + 1) % 4]
        else:
            new_dir = directions[(idx - 1) % 4]
        
        self.direction = new_dir

        x, y = self.head

        if self.direction == 'RIGHT':
            x += 1
        elif self.direction == 'LEFT':
            x -= 1
        elif self.direction == 'DOWN':
            y += 1
        elif self.direction == 'UP':
            y -= 1

        self.head = [x, y]

    def is_collision(self, pt = None):
        if pt is None:
            pt = self.head
        if pt[0] < 0 or pt[0] >= GRID_WIDTH or pt[1] < 0 or pt[1] >= GRID_HEIGHT:
            return True
        
        if pt in self.snake[1:]:
            return True
        
        return False

    def update_ui(self):
        self.display.fill(BACKGROUND)

        for index, pt in enumerate(self.snake):
            colour = HEAD if index == 0 else BODY
            rect = pygame.Rect(
                pt[0] * BLOCK_SIZE, pt[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE
            )
            pygame.draw.rect(self.display, colour, rect.inflate(-2, -2), border_radius=4)
        food_rect = pygame.Rect(
            self.food[0] * BLOCK_SIZE, self.food[1] * BLOCK_SIZE, BLOCK_SIZE, BLOCK_SIZE
        )
        pygame.draw.rect(self.display, FOOD, food_rect.inflate(-6, -6), border_radius=8)

        if self.render:
            pygame.display.flip()

    def play_step(self, action):
        self.frame_iteration += 1
        if self.render:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
        
        self.move(action)
        self.snake.insert(0, self.head[:])

        reward = 0
        game_over = False
        if self.is_collision() or self.frame_iteration > 100*len(self.snake):
            game_over = True
            reward = -10
            return reward, game_over, self.score
        
        if self.head == self.food:
            self.score += 1
            reward = 10
            self.spawn_food()
        else:
            self.snake.pop()

        self.update_ui()
        if self.clock is not None:
            self.clock.tick(self.fps)

        return reward, game_over, self.score
    
if __name__ == "__main__":
    game = SnakeGame()
    while True:
        keys = pygame.key.get_pressed()
        if keys[pygame.K_RIGHT]:  action = [0, 1, 0]  
        elif keys[pygame.K_LEFT]: action = [0, 0, 1]  
        else:                     action = [1, 0, 0] 

        reward, game_over, score = game.play_step(action)
        if game_over:
            print("Final Score: ", score)
            break