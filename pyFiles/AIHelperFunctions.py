import consts as c

def getAntState(map, ant):
    #Very complicated D,:
    #In future, po
    '''
    T = Top
    M = Middle
    B = Bottom
    L = Left
    R = Right

    Order:
    1. Ant % HP
    2. Ant % Hunger
    3. Ant's % width across the grid (to allow for variable grid sizes)
    4. Ant's % height across the grid (to allow for variable grid sizes)
    5. Colony % width across the grid (to allow for variable grid sizes)
    6. Colony % height across the grid (to allow for variable grid sizes)
    6.5. Enemy Colony % width across the grid (to allow for variable grid sizes) (IF SEEN)
    6.6. Enemy Colony % height across the grid (to allow for variable grid sizes) (IF SEEN)
    # WILL HAVE TO DECIDE HOW REALISTIC INCLUSION OF FOLLOWING IS. POSSIBLY OMIT AND IF AI TOO STUPID THEN ADD
    ext. Colony % HP
    ext. Colony % Hunger
    ext. Colony Total Food (ant food value + food stored value + larvae food value)
    ext. colony larvae?

    For TL -> TM -> TR -> ML -> MM -> MR -> BL -> BM -> BR
    7. TL Food Value
    8. TL Friendly Workers
    9. TL Friendly Soldiers
    10. TL Enemy Workers
    11. TL Enemy Soldiers
    12. TL Friendly Pheromone
    13. TL Enemy Pheromone
    '''
    state = []
    state.append(ant.hp/c.ANT_HP[ant.caste])
    state.append(ant.hunger/c.ANT_HUNGER)
    state.append(ant.pos[0]/c.SQUARE_LENGTH)
    state.append(ant.pos[1]/c.SQUARE_LENGTH)
    state.append(ant.col.pos[0]/c.SQUARE_LENGTH)
    state.append(ant.col.pos[1]/c.SQUARE_LENGTH)
    state.append(ant.heldFood/c.MAX_HELD_FOOD[ant.caste])
    state.append(0 if ant.caste == "worker" else 1)
    #DECIDE WHETHER TO INCLUDE ENEMY COLONY INFO
    state.append(ant.col.enemy.pos[0]/c.SQUARE_LENGTH if ant.col.enemy else -1)
    state.append(ant.col.enemy.pos[1]/c.SQUARE_LENGTH if ant.col.enemy else -1)
    #ENEMY COLONY POS INFO IF SEEN
    for x in range(ant.pos[0]-1, ant.pos[0]+2):
        for y in range(ant.pos[1]-1, ant.pos[1]+2):
            if x < 0 or x >= c.WIDTH or y < 0 or y >= c.HEIGHT:
                valid = 0
                friendlyWorkers = 0
                friendlySoldiers = 0
                enemyWorkers = 0
                enemySoldiers = 0
                foodValue = 0
                friendlyPheromoneType = -1
                enemyPheromoneType = -1
            else:
                valid = 1
                friendlyWorkers = 0
                friendlySoldiers = 0
                enemyWorkers = 0
                enemySoldiers = 0
                foodValue = 0
                friendlyPheromoneType = -1
                enemyPheromoneType = -1
                for entity in map[x][y]:
                    if entity.type == "ant":
                        if entity.caste == "worker":
                            if entity.col == ant.col:
                                friendlyWorkers += 1
                            else:
                                enemyWorkers += 1
                        else:
                            if entity.col == ant.col:
                                friendlySoldiers += 1
                            else:
                                enemySoldiers += 1
                    elif entity.type == "food":
                        foodValue = entity.value
                    elif entity.type == "pheromone":
                        if entity.col == ant.col.colour:
                            friendlyPheromoneType = entity.designation
                        else:
                            enemyPheromoneType = entity.designation
            state.append(valid)
            state.append(foodValue)
            state.append(friendlyWorkers)
            state.append(friendlySoldiers)
            state.append(enemyWorkers)
            state.append(enemySoldiers)
            state.append(friendlyPheromoneType)
            state.append(enemyPheromoneType)
    return state

def getColonyState(map, colony):
    '''
    Order:
    1. Colony % HP
    2. Colony % Hunger
    3. Colony Stored Food
    4. Colony Worker Larvae
    5. Colony Soldier Larvae
    6. Colony Worker Ants
    7. Colony Soldier Ants
    For TL -> TM -> TR -> ML -> MM -> MR -> BL -> BM -> BR
    7. TL Food Value
    8. TL Friendly Workers
    9. TL Friendly Soldiers
    10. TL Enemy Workers
    11. TL Enemy Soldiers
    12. TL Friendly Pheromone
    13. TL Enemy Pheromone
    '''
    state = []
    state.append(colony.hp/c.COLONY_HP)
    state.append(colony.hunger/c.COLONY_HUNGER)
    state.append(colony.foodStored)
    state.append(len(colony.larvae["worker"]))
    state.append(len(colony.larvae["soldier"]))
    state.append(len(colony.ants["worker"]))
    state.append(len(colony.ants["soldier"]))
    state.append(colony.heldFood)
    for x in range(colony.pos[0]-1, colony.pos[0]+2):
        for y in range(colony.pos[1]-1, colony.pos[1]+2):
            friendlyWorkers = 0
            friendlySoldiers = 0
            enemyWorkers = 0
            enemySoldiers = 0
            foodValue = 0
            friendlyPheromoneType = -1
            enemyPheromoneType = -1
            for entity in map[x][y]:
                if entity.type == "ant":
                    if entity.caste == "worker":
                        if entity.col == colony:
                            friendlyWorkers += 1
                        else:
                            enemyWorkers += 1
                    else:
                        if entity.col == colony:
                            friendlySoldiers += 1
                        else:
                            enemySoldiers += 1
                elif entity.type == "food":
                    foodValue = entity.value
                elif entity.type == "pheromone":
                    if entity.col == colony.colour:
                        friendlyPheromoneType = entity.designation
                    else:
                        enemyPheromoneType = entity.designation
            state.append(foodValue)
            state.append(friendlyWorkers)
            state.append(friendlySoldiers)
            state.append(enemyWorkers)
            state.append(enemySoldiers)
            state.append(friendlyPheromoneType)
            state.append(enemyPheromoneType)
    return state

def getAntReward(ant):
    reward = 0
    #reward += ant.hp/c.ANT_HP[ant.caste]
    #reward += ant.hunger/c.ANT_HUNGER
    reward += min(ant.col.foodValue - ant.col.lastFoodValue, c.MAX_ANT_FOOD_REWARD)*1000
    reward += ((ant.col.hp - ant.col.lastHP)/c.COLONY_HP)*c.ANT_COL_HP_REWARD_MULTIPLIER #MAKE INFO CARRY FROM LAST TURN SO DIFFERENCE IS MEASURED NOT ABSOLUTE HP ETC
    reward += ((ant.hp-ant.lastHP)/c.ANT_HP[ant.caste])*100
    #reward += colony time alive
    return reward

def getColReward(colony):
    reward = 0
    reward += min(colony.foodValue - colony.lastFoodValue, c.MAX_COL_FOOD_REWARD)*1000
    reward += ((colony.hp-colony.lastHP)/c.COLONY_HP)*c.COL_HP_REWARD_MULTIPLIER
    #reward += time alive
    return reward