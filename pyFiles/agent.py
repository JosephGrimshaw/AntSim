from collections import deque, defaultdict
import consts as c
import AIHelperFunctions as hf
import random
import torch
import numpy as np
import models

class Agent():
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
        self.memory.append((state, action, reward, nextState, done, hiddenState, ant_id))
            
    def getReward(self, entity, done):
        if self.organism == "ant":
            return hf.getAntReward(entity, done)
        return hf.getColReward(entity, done)
    
    def trainTurn(self, state, action, reward, nextState, done, entity):
        data = (state, action, reward, nextState, done, entity.hiddenState)
        #antDict = dict()
        #for dataSet in range(len(done)):

        antDict = {entity.ant_id: [data]}
        self.trainer.train(antDict)
        #entity.hiddenState = hiddenState[entity.ant_id]

    def trainGame(self, ants):
        self.epochs += 1
        if len(self.memory) > c.BATCH_SIZE:
            sample = random.sample(self.memory, c.BATCH_SIZE)
        else:
            sample = self.memory
        states, actions, rewards, nextStates, dones, hiddenStates, ant_ids = zip(*sample)

        antData = defaultdict(list)
        for i, ant_id in enumerate(ant_ids):
            if ant_id not in antData:
                antData[ant_id] = []
            antData[ant_id].append((states[i], actions[i], rewards[i], nextStates[i], dones[i], hiddenStates[i]))

        newHiddenStates = self.trainer.train(antData)
        ant_dict = {ant.ant_id: ant for ant in ants}
        for antID, newState in newHiddenStates.items():
            if antID in ant_dict:
                ant_dict[ant_id].hiddenState = newState

        if self.epochs % c.SAVE_INTERVAL == 0:
            torch.save(self.model.state_dict(), f=f"models/{self.organism}Model.pt")
    
    def getAction(self, gridState, globalState, entity):
        self.epsilon = max(c.INITIAL_EPSILON[self.organism] - self.epochs * c.EPSILON_DECAY[self.organism], c.MIN_EPSILON[self.organism])
        if random.randint(0, 200) < self.epsilon:
            action = random.choice(entity.allMoves)
        else:
            entity.hiddenState = (entity.hiddenState[0].to(models.device), entity.hiddenState[1].to(models.device))
            gridState1 = torch.tensor(gridState, dtype=torch.float).unsqueeze(0).to(models.device)  # Add batch dimension
            globalState1 = torch.tensor(globalState, dtype=torch.float).unsqueeze(0).to(models.device)
            action_index, entity.hiddenState = self.model(gridState1, globalState1, entity.hiddenState)
            action_index = action_index.argmax().item()
            action = entity.allMoves[action_index]
        return action