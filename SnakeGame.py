import pygame

import random
from enum import Enum
from collections import namedtuple

pygame.init()
font = pygame.font.SysFont('arial', 25)

class Direction(Enum):
        RIGHT = 1
        LEFT = 2
        UP = 3
        DOWN = 4

Point = namedtuple('Point', ['x', 'y'])

# rgb colors
WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)

BLOCK_SIZE = 10
SPEED = 20

class SnakeGame:
    def __init__(self, width=640, height=480):
        self.width = width
        self.height = height
        # init display
        self.display = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption('Snake? Snake?! Snaaaaaake!!')
        self.clock = pygame.time.Clock()

        # init game state
        self.reset()
    
    def reset(self):
        self.direction = Direction.RIGHT

        self.head = Point(self.width/2, self.height/2)
        self.snake = [self.head, Point(self.head.x-BLOCK_SIZE, self.head.y),
                      Point(self.head.x-2*BLOCK_SIZE, self.head.y)]
        
        self.score = 0
        self.food = None
        self._place_food()
        self.frameIteration = 0

    def _place_food(self):
        x = random.randint(0, (self.width-BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.height-BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def play_step(self):
        # 1. collect user input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_LEFT and self.direction != Direction.RIGHT:
                    self.direction = Direction.LEFT
                elif event.key == pygame.K_RIGHT and self.direction != Direction.LEFT:
                    self.direction = Direction.RIGHT
                elif event.key == pygame.K_UP and self.direction != Direction.DOWN:
                    self.direction = Direction.UP
                elif event.key == pygame.K_DOWN and self.direction != Direction.UP:
                    self.direction = Direction.DOWN
            
        # 2. move
        self._move(self.direction) # update the head
        self.snake.insert(0, self.head)

        # 3. check if game over
        gameOver = False
        if self._is_collision():
            gameOver = True
            return gameOver, self.score
        # 4. place new food or just move
        if self.head == self.food:
            self.score += 1
            self._place_food()
        else:
            self.snake.pop()

        # 5. update ui and clock
        self._update_ui()
        self.clock.tick(SPEED)

        # 6. return game over and score
        return gameOver, self.score

    def _is_collision(self):
        # hits boundary
        if self.head.x > self.width - BLOCK_SIZE or self.head.x < 0 or self.head.y > self.height - BLOCK_SIZE or self.head.y < 0:
            return True
        # hits self
        if self.head in self.snake[1:]:
            return True

        return False

    def _update_ui(self):
        self.display.fill(BLACK)

        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x+2, pt.y+2, BLOCK_SIZE* 0.6, BLOCK_SIZE * 0.6))

        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))

        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, direction):
        x = self.head.x
        y = self.head.y
        if direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif direction == Direction.UP:
            y -= BLOCK_SIZE
        elif direction == Direction.DOWN:
            y += BLOCK_SIZE

        self.head = Point(x, y)

if __name__ == '__main__':
    game = SnakeGame()

    #game loop
    while True:
        gameOver, score = game.play_step()

        if gameOver:
            break

    print('Final Score', score)

    pygame.quit()