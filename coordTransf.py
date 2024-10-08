
def transfCoord(coordinates):
    left, top, width, height = coordinates
    x1, y1 = left, top
    x2, y2 = left + width, top + height

    return x1,x2,y1,y2