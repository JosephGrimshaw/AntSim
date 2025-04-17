import extraFunctions as ef
import ant
import consts as c
import pheromone as p
import random

class Colony():
    def __init__(self, img, pos, workerImg, soldierImg, foodImg, colour, agent, antAgent):
        self.img = img
        self.pos = pos
        self.colour = colour
        self.agent = agent
        self.antAgent = antAgent
        self.lastState = None
        self.lastAction = None
        self.ants = {"soldier": [],
                     "worker": []}
        self.larvae = {"soldier": [],
                       "worker": []}
        self.foodImg = foodImg
        self.lastHP = c.COLONY_HP
        self.lastFoodValue = c.COLONY_HP
        self.foodStored = 0
        self.hunger = c.COLONY_HUNGER
        self.foodValue = c.COLONY_HP
        self.hp = c.COLONY_HP
        self.workerImg = workerImg
        self.soldierImg = soldierImg
        self.type = "colony"
        self.enemy = None
        self.allMoves = ["makeWorker", "makeSoldier", "eat", "skip"]
        for i in range(c.QUEEN_PHEROMONES):
            self.allMoves.append(f"layPheromone{i}")

    def draw(self, WIN):
        WIN.blit(self.img, (ef.squareToPixel(self.pos[0]), ef.squareToPixel(self.pos[1])))

    def makeAnt(self, caste, turns):
        self.larvae[caste].append(turns)

    def handleLarvae(self):
        newAnts = []
        newWorkers = 0
        newSoldiers = 0

        for idx in range(len(self.larvae["worker"])):
            if self.larvae["worker"][idx] == 0:
                newAnt = ant.Ant(self.workerImg, [self.pos[0], self.pos[1]], self, "worker", self.foodImg, self.antAgent)
                self.ants["worker"].append(newAnt)
                newAnts.append(newAnt)
                newWorkers += 1
            else:
                self.larvae["worker"][idx] -= 1


        for idx in range(len(self.larvae["soldier"])):
                if self.larvae["soldier"][idx] == 0:
                    newAnt = ant.Ant(self.soldierImg, [self.pos[0], self.pos[1]], self, "soldier", self.foodImg, self.antAgent)
                    self.ants["soldier"].append(newAnt)
                    newAnts.append(newAnt)
                    newSoldiers += 1
                else:
                    self.larvae["soldier"][idx] -= 1

        self.larvae["soldier"] = self.larvae["soldier"][newSoldiers:]
        self.larvae["worker"] = self.larvae["worker"][newWorkers:]
        
        return newAnts
    
    def updateFoodStored(self, square):
        foodStored = self.foodStored
        for entity in square:
            if entity.type == "food":
                if foodStored != entity.value:
                    self.foodStored = entity.value
                    self.foodValue += entity.value - foodStored
                return
    
    def takeTurn(self, map, done):
        #newState = self.agent.getState(map, self)
        #reward = self.agent.getReward(self)
        #if self.lastAction != None:
            #self.agent.trainTurn(self.lastState, self.allMoves.index(self.lastAction), reward, newState, done)
            #self.agent.record(self.lastState, self.allMoves.index(self.lastAction), reward, newState, done)
        #self.lastState = newState
        self.updateFoodStored(map[self.pos[0]][self.pos[1]])
        if self.hunger < c.COLONY_HP_DEGRADE_THRESHOLD_HUNGER:
            self.hp -= c.COLONY_HP_DEGRADE_HUNGER
            self.foodValue -= c.COLONY_HP_DEGRADE_HUNGER
        if self.hunger > c.COLONY_HP_HEAL_THRESHOLD_HUNGER:
            if self.hp + c.COLONY_HP_HEAL_HUNGER <= c.COLONY_HP:
                self.hp += c.COLONY_HP_HEAL_HUNGER
                self.foodValue += c.COLONY_HP_HEAL_HUNGER
        self.hunger -= c.COLONY_HUNGER_DEGRADE + len(self.larvae["worker"]) + len(self.larvae["soldier"])
        if self.hunger <= 0 or self.hp <= 0:
            if self.enemy:
                self.enemy.enemy = None
            return True, []
        #newMove = self.agent.getAction(newState, self)
        #self.lastAction = newMove
        newMove = random.choice(self.allMoves)
        self.lastHP = self.hp
        self.lastFoodValue = self.foodValue
        newObjs = self.handleLarvae()
        match newMove:
            case "makeWorker":
                if self.hunger >= c.NEW_WORKER_COST:
                    self.hunger -= c.NEW_WORKER_COST
                    self.makeAnt("worker", 5)
            case "makeSoldier":
                if self.hunger >= c.NEW_SOLDIER_COST:
                    self.hunger -= c.NEW_SOLDIER_COST
                    self.makeAnt("soldier", 10)
            case "eat":
                self.eat([entity for entity in map[self.pos[0]][self.pos[1]] if entity.type == "food"])
            case "skip":
                pass
            case _ if newMove.startswith("layPheromone"):
                idx = newMove.split("layPheromone")[1]
                idx = int(idx)
                newObjs.append(self.layPheromone(idx))
        return False, newObjs
    
    def eat(self, foodObjs):
        if not len(foodObjs):
            return
        for foodObj in foodObjs:
            extraHunger = c.COLONY_HUNGER - self.hunger
            eating = min(extraHunger, c.COLONY_MAX_EAT, foodObj.value)
            if eating:
                print("Colony Eating!")
                print("Hunger before: ", self.hunger)
                print("Food value before: ", foodObj.value)
            self.hunger += eating
            foodObj.value -= eating
            if eating:
                print("Hunger after:", self.hunger)
                print("Food value after: ", foodObj.value)

    def layPheromone(self, type):
        newPheromone = p.Pheromone(type, [self.pos[0], self.pos[1]], self.colour)
        return newPheromone