from micrograd_triton import MLP
from sklearn.datasets import make_moons
import numpy as np
from utils import plot_moons
import torch

np.random.seed(0)
torch.manual_seed(0)

device = 'cuda:0'
n_samples = 3000
X, Y = make_moons(n_samples=n_samples, random_state=1)
Y = Y * 2 - 1

Y = np.expand_dims(Y, axis=-1)

mlp = MLP(indim=2, outdim=1, hidden_dim=16, n_hidden_layers=1, activation=True, device=device)
mlp.load("torch_model")

preds = mlp(X)
print(preds)
preds = preds.cpu().detach().numpy()
correct = np.sign(Y) == np.sign(preds)
plot_moons(X, correct[:, 0])

print(f"accuracy:{np.mean(correct)}")

