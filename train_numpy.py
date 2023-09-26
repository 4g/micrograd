from micrograd_numpy import MLP
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


n_samples = 3000
X, Y = make_moons(n_samples=n_samples)
Y = Y * 2 - 1

Y = np.expand_dims(Y, axis=-1)

mlp = MLP(indim=2, outdim=1, hidden_dim=16, n_hidden_layers=2, activation=True)
print(mlp)

batch_size = 1
for epoch in range(100):
    losses = []
    correct = []
    n_batches = n_samples // batch_size
    for idx in range(n_batches):
        x = X[idx*batch_size:(idx+1)*batch_size]
        y = Y[idx*batch_size:(idx+1)*batch_size]
        ypred = mlp(x)
        loss = (-ypred*y + 1).relu()
        
        loss = loss.mean()

        # alpha = 1e-4
        # reg_loss = alpha * sum((p.norm() for p in mlp.params()))
        # total_loss = loss + reg_loss

        loss.backward()
        # print(loss)
        
        learning_rate = 1.0 - 0.9 * epoch / 100
        for param in mlp.params():
            param.data -= param.grad * learning_rate

        for param in mlp.params():
            param.grad = 0.0


    preds = mlp(X)
    correct = np.sign(Y) == np.sign(preds.data)
    print(f"Epoch:{epoch}, accuracy:{np.mean(correct)}")

    plot_moons(X, correct[:, 0])