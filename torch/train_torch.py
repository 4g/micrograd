from micrograd_torch import MLP, tensor
from sklearn.datasets import make_moons
import numpy as np
from utils import plot_moons

device = 'cuda:0'
n_samples = 3000
X, Y = make_moons(n_samples=n_samples)
Y = Y * 2 - 1

Y = np.expand_dims(Y, axis=-1)

mlp = MLP(indim=2, outdim=1, hidden_dim=16, n_hidden_layers=1, activation=True, device=device)
print(mlp)

batch_size = 16
for epoch in range(1):
    losses = []
    correct = []
    n_batches = n_samples // batch_size
    for idx in range(n_batches):
        x = X[idx*batch_size:(idx+1)*batch_size]
        y = Y[idx*batch_size:(idx+1)*batch_size]
        ypred = mlp(x)
        loss = (-ypred*tensor(y, device=device) + 1).relu()
        
        loss = loss.mean()
        loss.backward()
        
        learning_rate = 1.0 - 0.9 * epoch / 100
        for param in mlp.params():
            param.data -= param.grad * learning_rate

        for param in mlp.params():
            param.grad *= 0.0
        
        preds = mlp(X)
        preds = preds.cpu().detach().numpy()
        correct = np.sign(Y) == np.sign(preds)
        plot_moons(X, correct[:, 0])
    
    print(f"Epoch:{epoch}, accuracy:{np.mean(correct)}")

mlp.save("torch_model")