import pygame
import consts as c
import world as w
import pygameFunctions as pf
import colony as cn
import extraFunctions as ef
import agent as a
import models as m
import os
import torch
import ant
import cProfile

def main():
    #Initialize pygame
    if c.VISUALS:
        pygame.init()
        clock = pygame.time.Clock()
        WIN = pygame.display.set_mode((c.WIDTH, c.HEIGHT ))
        pygame.display.set_caption("Ant Simulator")

        #Load Images
        bgImage = pygame.image.load("./assets/grassBG.png").convert_alpha()
        colonyRedImage = pygame.image.load("./assets/colonyRed.png").convert_alpha()
        colonyBlueImage = pygame.image.load("./assets/colonyBlue.png").convert_alpha()
        soldierRedImage = pygame.image.load("./assets/soldierRed.png").convert_alpha()
        soldierBlueImage = pygame.image.load("./assets/soldierBlue.png").convert_alpha()
        workerRedImage = pygame.image.load("./assets/workerRed.png").convert_alpha()
        workerBlueImage = pygame.image.load("./assets/workerBlue.png").convert_alpha()
        foodImage = pygame.image.load("./assets/food.png").convert_alpha()

    else:
        #Set images to none to pass to objects
        bgImage = None
        colonyRedImage = None
        colonyBlueImage = None
        soldierRedImage = None
        soldierBlueImage = None
        workerRedImage = None
        soldierRedImage = None
        workerBlueImage = None
        foodImage = None

    #Create initial objects.
    #AI
    antModel = m.AntModel( c.ANT_INPUT_GRID_CHANNELS, c.ANT_INPUT_GLOBAL_SIZE,  c.ANT_OUTPUT_SIZE, c.ANT_HIDDEN_LAYERS, c.ANT_HIDDEN_NEURONES, c.ANT_RNN_HIDDEN_LAYERS, c.ANT_RNN_HIDDEN_NEURONES)
    antModel.to(m.device)
    #colModel = m.AntModel(c.COL_INPUT_SIZE, c.COL_OUTPUT_SIZE, c.COL_HIDDEN_LAYERS, c.COL_HIDDEN_NEURONES)
    if os.path.exists("./models/antModel.pt"):
        antModel.load_state_dict(torch.load("./models/antModel.pt"))
    #if os.path.exists("colModel.pt"):
        #colModel.load_state_dict(torch.load("colModel.pt"))
    antTrainer = m.Trainer(antModel, "ant")
    #colTrainer = m.Trainer(colModel, "colony")
    #colAgent = a.Agent(colModel, "colony", colTrainer)
    antAgent = a.Agent(antModel, "ant", antTrainer)

    for i in range(c.TOTAL_EPOCHS):
        print("Epoch: ", i)
        #Map
        map = []
        for x in range(c.WIDTH):
            row = []
            for y in range(c.HEIGHT):
                row.append([])
            map.append(row)

        world = w.World(bgImage)
        colonyRed = cn.Colony(colonyRedImage, [c.RED_COLONY_X, c.RED_COLONY_Y], workerRedImage, soldierRedImage, foodImage, "red", "colAgent", antAgent)
        colonyBlue = cn.Colony(colonyBlueImage, [c.BLUE_COLONY_X, c.BLUE_COLONY_Y], workerBlueImage, soldierBlueImage, foodImage, "blue", "colAgent", antAgent)
        colonyRed.enemy = colonyBlue
        colonyBlue.enemy = colonyRed
        map[c.RED_COLONY_X][c.RED_COLONY_Y].append(colonyRed)
        map[c.BLUE_COLONY_X][c.BLUE_COLONY_Y].append(colonyBlue)
        #############################
        ######## GAME LOOP ##########
        #############################

        run = True
        turns = 0
        while run:
            #Pygame content (optional)
            if c.VISUALS:
                #Ensure standard FPS
                clock.tick(c.FPS)
                #Handle all rendering
                world.draw(WIN)
                pf.drawAll(map, WIN)
                #Check pygame quit
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        run = False
                        if c.VISUALS:
                            antModel.save("./models/antModel.pt")
                            pygame.quit()
                pygame.display.flip()
            #Content
            map = ef.turn(map, foodImage, turns>=c.MAX_TURNS)
            turns += 1
            if turns >= c.MAX_TURNS or colonyRed not in map[c.RED_COLONY_X][c.RED_COLONY_Y] or colonyBlue not in map[c.BLUE_COLONY_X][c.BLUE_COLONY_Y]:
                run = False
                antAgent.trainGame(ant.Ant.allAnts)
                #colAgent.trainGame()
                antAgent.memory.clear()
                #colAgent.memory.clear()
            


        ##############################
        ######### END ################
        ##############################
    antModel.save("./models/antModel.pt")
    #colModel.save("colModel.pt")

#cProfile.run('main()')
main()