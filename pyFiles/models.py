import torch
import torch.nn as nn
import consts as c

#Ant model
class AntModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_size):
        super(AntModel, self).__init__()
        architecture = []
        architecture.append(nn.Linear(input_size, hidden_size))
        architecture.append(nn.ReLU())
        for _ in range(hidden_layers):
            architecture.append(nn.Linear(hidden_size, hidden_size))
            architecture.append(nn.ReLU())
        architecture.append(nn.Linear(hidden_size, output_size))
        self.compute = nn.Sequential(*architecture)

    def forward(self, x):
        return self.compute(x)
    
    def save(self, fileName):
        torch.save(self.state_dict(), f=fileName)

#Colony Model
class ColModel(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, hidden_size):
        super(ColModel, self).__init__()
        architecture = []
        architecture.append(nn.Linear(input_size, hidden_size))
        architecture.append(nn.ReLU())
        for _ in range(hidden_layers):
            architecture.append(nn.Linear(hidden_size, hidden_size))
            architecture.append(nn.ReLU())
        architecture.append(nn.Linear(hidden_size, output_size))
        self.compute = nn.Sequential(*architecture)

    def forward(self, x):
        return self.compute(x)
    
    def save(self, fileName):
        torch.save(self.state_dict(), f=fileName)

class Trainer():
    def __init__(self, model, type):
        self.model = model
        self.type = type
        if type == "ant":
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=c.ANT_LR)
            self.criterion = nn.MSELoss()
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=c.COL_LR)
            self.criterion = nn.MSELoss()
            
    def train(self, state, action, reward, nextState, done):
        state = torch.tensor(state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.float)
        reward = torch.tensor(reward, dtype=torch.float)
        nextState = torch.tensor(nextState, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            nextState = torch.unsqueeze(nextState, 0)
            done = (done, )
        
        pred = self.model(state)
        target = pred.clone()
        for i in range(len(done)):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                target[i][int(action[i])] = reward[i] + c.GAMMA[self.type] * torch.max(self.model(nextState[i])).item()

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()