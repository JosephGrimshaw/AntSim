from collections import deque
import consts as c
import AIHelperFunctions as hf
import random
import torch
import numpy as np

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
            state = np.array(hf.getAntState(map, entity), dtype=int)
            return state
        state = np.array(hf.getColonyState(map, entity), dtype=int)
        return state
    
    def record(self, state, action, reward, nextState, done):
        self.memory.append((state, action, reward, nextState, done))
            
    def getReward(self, entity):
        if self.organism == "ant":
            return hf.getAntReward(entity)
        return hf.getColReward(entity)
    
    def trainTurn(self, state, action, reward, nextState, done):
        self.trainer.train(state, action, reward, nextState, done)

    def trainGame(self):
        self.epochs += 1
        if len(self.memory) > c.BATCH_SIZE:
            sample = random.sample(self.memory, c.BATCH_SIZE)
        else:
            sample = self.memory
        states, actions, rewards, nextStates, dones = zip(*sample)
        self.trainer.train(states, actions, rewards, nextStates, dones)
        if self.epochs % c.SAVE_INTERVAL == 0:
            torch.save(self.model.state_dict(), f=f"models/{self.organism}Model.pt")
    
    def getAction(self, state, entity):
        self.epsilon = max(c.INITIAL_EPSILON[self.organism] - self.epochs * c.EPSILON_DECAY[self.organism], c.MIN_EPSILON[self.organism])
        if random.randint(0, 200) < self.epsilon:
            action = random.choice(entity.allMoves)
        else:
            state1 = torch.tensor(state, dtype=torch.float).unsqueeze(0)  # Add batch dimension
            action_index = self.model(state1).argmax().item()
            action = entity.allMoves[action_index]
        return action