from collections import deque, defaultdict
import consts as c
import AIHelperFunctions as hf
import random
import torch
import numpy as np

class Agent():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    def __init__(self, model, organism, trainer):
        self.model = model
        self.trainer = trainer
        self.epochs = 0
        self.epsilon = 0
        self.memory = deque(maxlen=c.MAX_DEQUE_MEMORY)
        self.organism = organism
    
    def getState(self, map, entity):
        if self.organism == "ant":
            state = hf.getAntGridState(map, entity)
            globalState = hf.getAntGlobalState(entity)
            return [state, globalState]
        state = hf.getColonyState(map, entity)
        return state
    
    def record(self, state, action, reward, nextState, done, hiddenState, ant_id):
        self.memory.append((state[0], state[1], action, reward, nextState[0], nextState[1], done, hiddenState[0], hiddenState[1], ant_id))
            
    def getReward(self, entity, done):
        if self.organism == "ant":
            return hf.getAntReward(entity, done)
        return hf.getColReward(entity, done)
    
    def trainTurn(self, state, action, reward, nextState, done, entity):
        #data = (state, action, reward, nextState, done, entity.hiddenState)
        #antDict = dict()
        #for dataSet in range(len(done)):

        #antDict = {entity.ant_id: [data]}
        self.trainer.train(state[0], state[1], action, reward, nextState[0], nextState[1], done, entity.hiddenState[0], entity.hiddenState[1])
        #entity.hiddenState = hiddenState[entity.ant_id]

    def trainGame(self, ants):
        self.epochs += 1
        if len(self.memory) > c.BATCH_SIZE:
            sample = random.sample(self.memory, c.BATCH_SIZE)
        else:
            sample = self.memory
        gridState, globalState, actions, rewards, nextGridStates, nextGlobalStates, dones, hiddenStatesH, hiddenStatesC, ant_ids = zip(*sample)
        hiddenStatesH = torch.stack(hiddenStatesH, dim=0).squeeze(2)
        hiddenStatesC = torch.stack(hiddenStatesC, dim=0).squeeze(2)
        #print("Before permute: ",  hiddenStatesH.shape, "\n", hiddenStatesC.shape)
        hiddenStatesH = hiddenStatesH.permute(1, 0, 2).contiguous()  # Reshape to (num_layers, batch_size, hidden_size)
        hiddenStatesC = hiddenStatesC.permute(1, 0, 2).contiguous()  # Reshape to (num_layers, batch_size, hidden_size)
        #print("After permute: ",  hiddenStatesH.shape, "\n", hiddenStatesC.shape)
        '''
        antData = defaultdict(list)
        for i, ant_id in enumerate(ant_ids):
            if ant_id not in antData:
                antData[ant_id] = []
            antData[ant_id].append((states[i], actions[i], rewards[i], nextStates[i], dones[i], hiddenStates[i]))
        '''
        self.trainer.train(gridState, globalState, actions, rewards, nextGridStates, nextGlobalStates, dones, hiddenStatesH, hiddenStatesC)
        '''
        ant_dict = {ant.ant_id: ant for ant in ants}
        for antID, newState in newHiddenStates.items():
            if antID in ant_dict:
                ant_dict[ant_id].hiddenState = newState
        '''
        if self.epochs % c.SAVE_INTERVAL == 0:
            torch.save(self.model.state_dict(), f=f"models/{self.organism}Model.pt")
    
    def getAction(self, gridState, globalState, entity):
        self.epsilon = max(c.INITIAL_EPSILON[self.organism] - self.epochs * c.EPSILON_DECAY[self.organism], c.MIN_EPSILON[self.organism])
        if random.randint(0, 200) < self.epsilon:
            action = random.choice(entity.allMoves)
        else:
            entity.hiddenState = (entity.hiddenState[0].to(Agent.device), entity.hiddenState[1].to(Agent.device))
            gridState1 = torch.tensor(gridState, dtype=torch.float).unsqueeze(0).to(Agent.device)  # Add batch dimension
            globalState1 = torch.tensor(globalState, dtype=torch.float).unsqueeze(0).to(Agent.device)
            action_index, entity.hiddenState = self.model(gridState1, globalState1, entity.hiddenState)
            action_index = action_index.argmax().item()
            action = entity.allMoves[action_index]
        return action