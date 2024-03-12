import torch
import random
import numpy as np
from collections import deque
from game import SnakeGameAI, Direction, Point

MAX_MEMORY = 100000
BATCH_SIZE = 1000
LEARNING_RATE = 0.001

class Agent:
    def __init__(self):
        self.numGames = 0
        self.epsilon = 0 # controls randomness
        self.gamma = 0 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft() if full
        self.model = None # TODO
        self.trainer = None # TODO
        # TODO: model, trainer

    def get_state(self, game):
        # game state is a list of 11 values
        head = game.snake[0]
        point_l = Point(head.x - 20, head.y)
        point_r = Point(head.x + 20, head.y)
        point_u = Point(head.x, head.y - 20)
        point_d = Point(head.x, head.y + 20)
        
        dir_l = game.direction == Direction.LEFT
        dir_r = game.direction == Direction.RIGHT
        dir_u = game.direction == Direction.UP
        dir_d = game.direction == Direction.DOWN

        state = [
            # Danger straight
            (dir_r and game.is_collision(point_r)) or 
            (dir_l and game.is_collision(point_l)) or 
            (dir_u and game.is_collision(point_u)) or 
            (dir_d and game.is_collision(point_d)),

            # Danger right
            (dir_u and game.is_collision(point_r)) or 
            (dir_d and game.is_collision(point_l)) or 
            (dir_l and game.is_collision(point_u)) or 
            (dir_r and game.is_collision(point_d)),

            # Danger left
            (dir_d and game.is_collision(point_r)) or 
            (dir_u and game.is_collision(point_l)) or 
            (dir_r and game.is_collision(point_u)) or 
            (dir_l and game.is_collision(point_d)),
            
            # Move direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,
            
            # Food location 
            game.food.x < game.head.x,  # food left
            game.food.x > game.head.x,  # food right
            game.food.y < game.head.y,  # food up
            game.food.y > game.head.y  # food down
            ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, nextState, gameOver):
        self.memory.append((state, action, reward, nextState, gameOver))

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            miniSample = random.sample(self.memory, BATCH_SIZE) # returns a random list of 1000 tuples from memory
        else:
            miniSample = self.memory

        states, actions, rewards, nextStates, gameOvers = zip(*miniSample)
        self.trainer.train_step(states, actions, rewards, nextStates, gameOvers)

    def train_short_memory(self, state, action, reward, nextState, gameOver):
        self.trainer.train_step(state, action, reward, nextState, gameOver)

    def get_action(self, state):
        # random moves: tradeoff b/w exploration and exploitation
        self.epsilon = 80 - self.numGames # the more games, the less random moves
        finalMove = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
           move = random.randint(0, 2)
           finalMove[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.model.predict(state0)
            move = torch.argmax(prediction).item()
            finalMove[move] = 1

        return finalMove

def train():
    plotScores = []
    plotMeanScores = []
    totalScore = 0
    record = 0
    agent = Agent()
    game = SnakeGameAI()

    while True:
        # get old state
        oldState = agent.get_state(game)

        # get next move
        nextMove = agent.get_action(oldState)

        # perform move and get new state
        reward, gameOver, score = game.play_step(nextMove)
        newState = agent.get_state(game)

        # train short memory (1 step)
        agent.train_short_memory(oldState, nextMove, reward, newState, gameOver)

        #remember
        agent.remember(oldState, nextMove, reward, newState, gameOver)

        if gameOver:
            # train long memory (entire game)
            game.reset()
            agent.numGames += 1
            agent.train_long_memory()

            if score > record:
                record = score
                # TODO: agent.model.save()

            print('Game ', agent.numGames, 'Score ', score, 'Record ', record)

            #TODO: plot stats

if __name__ == '__main__':
    train()