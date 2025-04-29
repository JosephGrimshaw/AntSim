import torch
import torch.nn as nn
import consts as c

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#Ant model
class AntModel(nn.Module):
    def __init__(self, input_grid_channels, input_global_size, output_size, hidden_layers, hidden_size, rnn_hidden_layers, rnn_layer_neurones):
        super(AntModel, self).__init__()
        self.global_fc = nn.Sequential(
            nn.Linear(input_global_size, 32),
            nn.ReLU())
        self.cnn = nn.Sequential(
            nn.Conv2d(input_grid_channels, 16, kernel_size = 2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size= 2),
            nn.ReLU(),
            nn.Flatten()
        )
        self.rnn = nn.LSTM(64, rnn_layer_neurones, rnn_hidden_layers, batch_first=True)
        architecture = []
        architecture.append(nn.Linear(rnn_layer_neurones, hidden_size))
        architecture.append(nn.ReLU())
        for _ in range(hidden_layers):
            architecture.append(nn.Linear(hidden_size, hidden_size))
            architecture.append(nn.ReLU())
        architecture.append(nn.Linear(hidden_size, output_size))
        self.combined = nn.Sequential(*architecture)

    def forward(self, gridX, globalX, hiddenState):
        gridX = self.cnn(gridX)  # Shape: (batch_size, 64)
        if gridX.shape[1] == 1:
            gridX = gridX.squeeze(-1).unsqueeze(0)  # Add sequence dimension
        #globalX = globalX.unsqueeze(1).squeeze(0)
        globalX = self.global_fc(globalX)  # Shape: (batch_size, 32)
        if len(globalX.shape) < 2:
            globalX = globalX.unsqueeze(0)
        x = torch.cat((gridX, globalX), dim=-1)  # Concatenate along feature dimension
        x = x.unsqueeze(1)                      # Should be (batch, seq_len=1, 96)
        x, hiddenState = self.rnn(x, hiddenState)  # Pass through RNN

        #x = x.squeeze(0)  # Remove sequence dimension

        return self.combined(x), hiddenState
    
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

    def trainSingle(self, state, action, reward, nextState, done, hiddenState):
        gridState = torch.tensor(state[0], dtype=torch.float).unsqueeze(0).to(device)
        if len(gridState.shape) == 3:
            gridState = gridState.unsqueeze(1)
        globalState = torch.tensor(state[1], dtype=torch.float).unsqueeze(0).to(device)
        action = torch.tensor([action], dtype=torch.long).to(device)
        reward = torch.tensor([reward], dtype=torch.float).to(device)
        nextGridState = torch.tensor(nextState[0], dtype=torch.float).unsqueeze(0).to(device)
        nextGlobalState = torch.tensor(nextState[1], dtype=torch.float).unsqueeze(0).to(device)

        #hiddenState = (hiddenState[0].detach().clone().to(device), hiddenState[1].detach().clone().to(device))
        hiddenState = (hiddenState[0].to(device), hiddenState[1].to(device))
        pred, hiddenState = self.model(gridState, globalState, hiddenState)
        target = pred.clone().detach()

        hiddenState = (hiddenState[0].detach().clone(), hiddenState[1].detach().clone())

        if done:
                target[0][0][action.item()] = reward.item()
        else:
            newState, _ = self.model(nextGridState, nextGlobalState, hiddenState)
            target[0][0][action.item()] = reward.item() + c.GAMMA[self.type] * torch.max(newState).item()
        
        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        self.optimizer.step()

        return hiddenState
            
    def train(self, antData):
        newHiddenStates = {}

        for antID, samples in antData.items():
            state, action, reward, nextState, done, hiddenState = samples[-1]
            assert hiddenState is not None, f"Hidden state is None for antID {antID}"
            newHidden = self.trainSingle(state, action, reward, nextState, done, hiddenState)
            newHiddenStates[antID] = newHidden
        
        return newHiddenStates
        '''
        gridState = torch.tensor(state[0], dtype=torch.float)
        globalState = torch.tensor(state[1], dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)
        nextGridState = torch.tensor(nextState[0], dtype=torch.float)
        nextGlobalState = torch.tensor(nextState[1], dtype=torch.float)

        if len(globalState.shape) == 1:
            print("Hi")
            gridState = torch.unsqueeze(gridState, 0)
            globalState = torch.unsqueeze(globalState, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            nextGridState = torch.unsqueeze(nextGridState, 0)
            nextGlobalState = torch.unsqueeze(nextGlobalState, 0)
            done = (done, )
        else:
            print("Other Hi")
            gridState = torch.unsqueeze(gridState, 0).unsqueeze(0).squeeze(1)
            globalState = torch.unsqueeze(globalState, 0).unsqueeze(1)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            nextGridState = torch.unsqueeze(nextGridState, 0).unsqueeze(0).unsqueeze(1)
            nextGlobalState = torch.unsqueeze(nextGlobalState, 0).unsqueeze(1)
        pred, hiddenState = self.model(gridState, globalState, hiddenState)
        target = pred.detach().clone()
        hiddenState = (hiddenState[0].detach(), hiddenState[1].detach())
        for i in range(len(done)):
            if done[i]:
                target[i][action[i]] = reward[i]
            else:
                newState, _ = self.model(nextGridState[i], nextGlobalState[i], hiddenState)
                target[i][int(action[i])] = reward[i] + c.GAMMA[self.type] * torch.max(newState).item()

        assert hiddenState[0].shape[1] == len(done), "Hidden state batch size does not match input batch size"

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        torch.autograd.set_detect_anomaly(True)
        loss.backward()
        self.optimizer.step()
        return hiddenState
        '''