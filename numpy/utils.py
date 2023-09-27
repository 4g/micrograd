import numpy as np

def plot_moons(X, y):
    w = 20
    np.set_printoptions(linewidth=100000, threshold=100000)
    image = np.ones((w, w), dtype=np.uint8)*8

    for x, y_ in zip(X, y):
        xi = x[0] * w/4 + w/2 - 1
        yi = x[1] * w/4 + w/2 - 1
        image[int(xi), int(yi)] = y_

    image = str(image)
    image = image.replace("8"," ")
    image = image.replace("0", "x")
    image = image.replace("1", ".")
    print(image, end='\r')

    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'

    for l in range(w-1):
        print(LINE_UP, end=LINE_CLEAR)
