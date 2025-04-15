import consts as c

class Food():
    def __init__(self, img, value, pos):
        self.img = img
        self.value = value
        self.pos = pos
        self.type = "food"

    def draw(self, WIN):
        WIN.blit(self.img, (int((self.pos[0]/c.SQUARE_LENGTH)*c.WIDTH), int((self.pos[1]/c.SQUARE_LENGTH)*c.HEIGHT)))
    
    def takeTurn(self, map):
        if self.value <= 0:
            return True, []
        if (any(entity.type == "colony" for entity in map[self.pos[0]][self.pos[1]])):
            return False, []
        self.value -= 1
        return False, []
        