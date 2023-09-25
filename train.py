from micrograd_python import MLP
from sklearn.datasets import make_moons
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

X, Y = make_moons(n_samples=300)
Y = Y * 2 - 1

mlp = MLP(indim=2, outdim=1, hidden_dim=16, n_hidden_layers=1, activation=True)
print(mlp)
for epoch in range(100):
    losses = []
    correct = []
    for idx, (x, y) in enumerate(zip(X, Y)):
        ypred = mlp(x)[0]
        loss = (1 + -y*ypred).relu()
        losses.append(loss)
        correct.append(np.sign(ypred.data) == np.sign(y))
        # print("==",ypred, y, loss)

    loss = sum(losses) * (1. / len(losses))
    alpha = 1e-4
    reg_loss = alpha * sum((p * p for p in mlp.params()))
    total_loss = loss + reg_loss

    total_loss.backward()

    learning_rate = 1.0 - 0.9 * epoch / 100
    for param in mlp.params():
        param.data -= param.grad * learning_rate

    for param in mlp.params():
        param.grad = 0.0

    # plot_moons(X, correct)