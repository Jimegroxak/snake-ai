import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name='model.pth'):
        modelFolderPath = './model'
        if not os.path.exists(modelFolderPath):
            os.makedirs(modelFolderPath)

        file_name = os.path.join(modelFolderPath, file_name)
        torch.save(self.state_dict(), file_name)

class QTrainer:
    def __init__(self, model, learningRate, gamma):
        self.learningRate = learningRate
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learningRate)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, nextState, gameOver):
        state = torch.tensor(state, dtype=torch.float)
        nextState = torch.tensor(nextState, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        # multiple states, (n, x)

        if len(state.shape) == 1:
            # only one state, need (1, x)
            state = torch.unsqueeze(state, 0)
            nextState = torch.unsqueeze(nextState, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            gameOver = (gameOver, )

        # 1: predicted Q values with current state, one for each action
        pred = self.model(state)

        target = pred.clone()
        for i in range(len(gameOver)):
            Q_new = reward[i]
            if not gameOver[i]:
                Q_new = reward[i] + self.gamma * torch.max(self.model(nextState[i]))
            
            target[i][torch.argmax(action).item()] = Q_new

        # 2: Q_new = r + y * max(next predicted Q value) -> only do if not done before

        # I think this updates the bias between neurons?
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()

        self.optimizer.step()


