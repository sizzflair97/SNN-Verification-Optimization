def compare(x, y):
    xx, yy = int(x.name().split('_')[-1]), int(y.name().split('_')[-1])
    return xx-yy