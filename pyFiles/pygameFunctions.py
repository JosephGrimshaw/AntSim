def drawAll(map, WIN):
    for row in map:
        for square in row:
            for entity in square:
                entity.draw(WIN)