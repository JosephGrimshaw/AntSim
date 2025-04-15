import extraFunctions as ef
import ant
import random
import consts as c
import pheromone as p

class Colony():
    def __init__(self, img, pos, workerImg, soldierImg, foodImg, colour):
        self.img = img
        self.pos = pos
        self.colour = colour
        self.ants = {"soldier": [],
                     "worker": []}
        self.larvae = {"soldier": [],
                       "worker": []}
        self.foodImg = foodImg
        self.hunger = c.COLONY_HUNGER
        self.hp = c.COLONY_HP
        self.workerImg = workerImg
        self.soldierImg = soldierImg
        self.type = "colony"
        self.enemy = None
        self.allMoves = ["makeWorker", "makeSoldier", "skip"]
        for i in range(c.ANT_PHEROMONES, c.QUEEN_PHEROMONES):
            self.allMoves.append(["layPheromone", i])

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
                newAnt = ant.Ant(self.workerImg, [self.pos[0], self.pos[1]], self, "worker", self.foodImg)
                self.ants["worker"].append(newAnt)
                newAnts.append(newAnt)
                newWorkers += 1
            else:
                self.larvae["worker"][idx] -= 1


        for idx in range(len(self.larvae["soldier"])):
                if self.larvae["soldier"][idx] == 0:
                    newAnt = ant.Ant(self.soldierImg, [self.pos[0], self.pos[1]], self, "soldier", self.foodImg)
                    self.ants["soldier"].append(newAnt)
                    newAnts.append(newAnt)
                    newSoldiers += 1
                else:
                    self.larvae["soldier"][idx] -= 1

        self.larvae["soldier"] = self.larvae["soldier"][newSoldiers:]
        self.larvae["worker"] = self.larvae["worker"][newWorkers:]
        
        return newAnts
            
    
    def takeTurn(self, map):
        if self.hunger < c.COLONY_HP_DEGRADE_THRESHOLD_HUNGER:
            self.hp -= c.COLONY_HP_DEGRADE_HUNGER
        self.hunger -= c.COLONY_HUNGER_DEGRADE + len(self.larvae["worker"]) + len(self.larvae["soldier"])
        if self.hunger <= 0 or self.hp <= 0:
            self.enemy.enemy = None
            return True, []
        newMove = random.choice(self.allMoves)
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
                self.eat(any(entity.type == "food" for entity in map[self.pos[0]][self.pos[1]]))
            case "skip":
                pass
            case _:
                newObjs.append(self.layPheromone(newMove[1]))
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
        return [newPheromone]