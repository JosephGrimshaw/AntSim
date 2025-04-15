import consts as c
import extraFunctions as ef
import pygame

class Pheromone():
    def __init__(self, designation, pos, col):
        self.type = "pheromone"
        self.designation = designation
        self.pos = pos
        self.col = col
        self.colourConst = designation * c.PHEROMONE_COLOUR_MULTIPLIER
        self.pixelPos = [ef.squareToPixel(pos[0])+((c.PHEROMONES_SQUARE_LENGTH%(self.designation+1))*c.PHEROMONES_PIXEL_WIDTH)-c.PHEROMONES_PIXEL_WIDTH, ef.squareToPixel(pos[1])+((c.PHEROMONES_SQUARE_LENGTH//(self.designation+1))*c.PHEROMONES_PIXEL_WIDTH)-c.PHEROMONES_PIXEL_WIDTH]
        if designation > c.ANT_PHEROMONES-1:
            self.duration = c.QUEEN_PHEROMONE_DURATION
            self.attach = True
        else:
            self.duration = c.ANT_PHEROMONE_DURATION
    
    def takeTurn(self, map):
        self.duration -= 1
        if self.duration <= 0:
            return True, None
        for entity in map[self.pos[0]][self.pos[1]]:
            if entity.type == "pheromone" and entity != self:
                if self.duration < entity.duration:
                    return True, None
        return False, None
    
    def draw(self, WIN):
        rect = pygame.Rect(self.pixelPos[0], self.pixelPos[1], c.PHEROMONES_PIXEL_WIDTH, c.PHEROMONES_PIXEL_WIDTH)
        pygame.Surface.fill(WIN, (self.colourConst, 255-self.colourConst, c.PHEROMONE_STATIC_COLOUR[self.col]), rect)