import pygame
import consts as c
import world as w
import pygameFunctions as pf
import colony as cn
import extraFunctions as ef

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
#Map
map = []
for x in range(c.WIDTH):
    row = []
    for y in range(c.HEIGHT):
        row.append([])
    map.append(row)

world = w.World(bgImage)
colonyRed = cn.Colony(colonyRedImage, [c.RED_COLONY_X, c.RED_COLONY_Y], workerRedImage, soldierRedImage, foodImage, "red")
colonyBlue = cn.Colony(colonyBlueImage, [c.BLUE_COLONY_X, c.BLUE_COLONY_Y], workerBlueImage, soldierBlueImage, foodImage, "blue")
colonyRed.enemy = colonyBlue
colonyBlue.enemy = colonyRed
map[c.RED_COLONY_X][c.RED_COLONY_Y].append(colonyRed)
map[c.BLUE_COLONY_X][c.BLUE_COLONY_Y].append(colonyBlue)
#############################
######## GAME LOOP ##########
#############################

run = True
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
        pygame.display.flip()
    #Content
    map = ef.turn(map, foodImage)
    


##############################
######### END ################
##############################

if c.VISUALS:
    pygame.quit()