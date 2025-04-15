import consts as c
import food as f
import random

def squareToPixel(square):
    pixel = int((square/c.SQUARE_LENGTH) * c.WIDTH)
    return pixel

#############################################
######### IMPORTANT #########################
#############################################

def turn(map, foodImg, done):
    changes = {
        "add": [],
        "remove": []
    }
    map = bundleGameTurn(map, changes, foodImg, done)
    return map

def bundleGameTurn(map, changes, foodImg, done):
    changes = handleFood(map, changes, foodImg)       
    changes = handleMap(map, changes, done)
    map = changesEffector(map, changes)
    return map

def handleMap(map, changes, done):
    for row in map:
        for square in row:
            for entity in square[:]:
                oldPos = entity.pos[:]
                delete, new = entity.takeTurn(map, done)
                if new:
                    for newEntity in new:
                        changes["add"].append([newEntity, newEntity.pos])
                if delete:
                    changes["remove"].append([entity, oldPos])
    return changes

def changesEffector(map, changes):
    for change in changes["add"]:
        map[change[1][0]][change[1][1]].append(change[0])
    for change in changes["remove"]:        
        map[change[1][0]][change[1][1]].remove(change[0])
    return map

def handleFood(map, changes, foodImg):
    for i in range(c.MAX_ADDED_FOOD):
        if random.randint(0, c.FOOD_RATE) == 1:
            x = random.randint(0, c.SQUARE_LENGTH-1)
            y = random.randint(0, c.SQUARE_LENGTH-1)
            makeNew = True
            for entity in map[x][y]:
                if entity.type == "food":
                    entity.value += random.randint(c.MIN_FOOD_VALUE, c.MAX_FOOD_VALUE)
                    makeNew = False
            if makeNew:
                changes["add"].append([f.Food(foodImg, random.randint(1, 100), (x, y)), [x,  y]])
    return changes