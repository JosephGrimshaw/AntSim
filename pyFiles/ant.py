import extraFunctions as ef
import random
import consts as c
import food as f
import pheromone as p

class Ant():
    def __init__(self, img, pos, col, caste, foodImg):
        self.img = img
        self.pos = pos
        self.col = col
        self.foodImg = foodImg
        self.type = "ant"
        self.caste = caste
        self.hunger = c.ANT_HUNGER
        self.heldFood = 0
        self.allMoves = ["left", "right", "up", "down", "pickUpFood", "eatFood", "dropFood", "attackAnt", "attackColony", "skip"]
        for i in range(c.ANT_PHEROMONES):
            self.allMoves.append(["layPheromone", i])
        if caste == "worker":
            self.dmg = c.WORKER_DMG
            self.hp = c.WORKER_HP
        elif caste == "soldier":
            self.dmg = c.SOLDIER_DMG
            self.hp = c.SOLDIER_HP

    def draw(self, WIN):
        WIN.blit(self.img, (ef.squareToPixel(self.pos[0]), ef.squareToPixel(self.pos[1])))

    def takeTurn(self, map):
        if self.hunger <= c.ANT_HP_DEGRADE_THRESHOLD_HUNGER:
            self.hp -= c.ANT_HP_DEGRADE_HUNGER
        self.hunger -= c.ANT_HUNGER_DEGRADE
        if self.hp <= 0 or self.hunger <= 0:
            if self in self.col.ants[self.caste]:
                self.col.ants[self.caste].remove(self)
            else:
                pass
            #Delete self
            foodObj = f.Food(self.foodImg, c.DEAD_FOODS[self.caste], [self.pos[0], self.pos[1]])
            return True, [foodObj]
        newMove = random.choice(self.allMoves)
        match newMove:
            case "left":
                return True, self.move("left")
            case "right":
                return True, self.move("right")
            case "up":
                return True, self.move("up")
            case "down":
                return True, self.move("down")
            case "pickUpFood":
                self.pickUpFood(map)
                return False, None
            case "eatFood":
                self.eatFood()
                return False, None
            case "dropFood":
                return False, self.dropFood()
            case "attackAnt":
                self.attackAnt(map)
                return False, None
            case "attackColony":
                self.attackColony(map)
                return False, None
            case "skip":
                return False, None
            case _:
                return False, self.layPheromone(newMove[1])
    
    def pickUpFood(self, map):
        for entity in map[self.pos[0]][self.pos[1]]:
            if entity.type == "food":
                if entity.value > 0 and c.MAX_HELD_FOOD[self.caste] > self.heldFood:
                    if entity.value > c.MAX_HELD_FOOD[self.caste]-self.heldFood:
                        added = c.MAX_HELD_FOOD[self.caste]-self.heldFood
                        self.heldFood = c.MAX_HELD_FOOD[self.caste]
                        entity.value -= added
                        return None
                    self.heldFood += entity.value
                    entity.value = 0
                    return None
                return None
        return None
    
    def dropFood(self):
        if not self.heldFood:
            return None
        foodObj = f.Food(self.foodImg, self.heldFood, [self.pos[0], self.pos[1]])
        self.heldFood = 0
        return [foodObj]
    
    def eatFood(self):
        emptyHunger = c.ANT_HUNGER - self.hunger
        eating = min(emptyHunger, self.heldFood)
        self.heldFood -= eating
        self.hunger += eating


    def move(self, dir):
        match dir:
            case "left":
                if self.pos[0] > 0:
                    self.pos[0] -= 1
                else:
                    self.col.ants[self.caste].remove(self)
                    return None
            case "right":
                if self.pos[0] < c.SQUARE_LENGTH -1:
                    self.pos[0] += 1
                else:
                    self.col.ants[self.caste].remove(self)
                    return None
            case "up":
                if self.pos[1] > 0:
                    self.pos[1] -= 1
                else:
                    self.col.ants[self.caste].remove(self)
                    return None
            case "down":
                if self.pos[1] < c.SQUARE_LENGTH -1:
                    self.pos[1] += 1
                else:
                    self.col.ants[self.caste].remove(self)
                    return None
        #Check if ant is within bounds of colony
        if 0 <= self.pos[0] < c.SQUARE_LENGTH and 0 <= self.pos[1] < c.SQUARE_LENGTH:
            return [self]
        self.col.ants[self.caste].remove(self)
        return None

    def attackAnt(self, map):
        if self.heldFood:
            return
        enemyAnts = []
        #In future maybe add bonus dmg for each friendly ant in space as well?
        for entity in map[self.pos[0]][self.pos[1]]:
            if entity.type == "ant" and entity.col != self.col:
                enemyAnts.append(entity)
        if not len(enemyAnts):
            return
        random.shuffle(enemyAnts)
        enemyAnts[0].hp -= self.dmg
    
    def attackColony(self, map):
        if self.heldFood:
            return
        for entity in map[self.pos[0]][self.pos[1]]:
            if entity.type == "colony" and entity != self.col:
                entity.hp -= self.dmg
                return
            
    def layPheromone(self, type):
        newPheromone = p.Pheromone(type, [self.pos[0], self.pos[1]], self.col.colour)
        return [newPheromone]