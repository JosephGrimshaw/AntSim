import consts as c
import pygame

class World():
    def __init__(self, bgImage=None):
        self.bgImage = bgImage

    def draw(self, WIN):
        #Background
        WIN.blit(self.bgImage, (0,0))
        for x in range(0, c.WIDTH, c.SQUARE_PIXEL_SIZE):
            pygame.draw.line(WIN, (255,255,255), (x, 0), (x, c.HEIGHT), 1)
            pygame.draw.line(WIN, (255,255,255), (0, x), (c.WIDTH, x), 1)
        