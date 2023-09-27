from torch import Tensor
import numpy as np


class Layer:
    ID = 0

    def __init__(self, indim, outdim, activation=True, name=None):
        Layer.ID += 1

        self.id = Layer.ID
        self.name = name if name else f"LAYER_{Layer.ID}"
        self.indim = indim
        self.outdim = outdim
        self.activation = activation
        self.weights = Tensor(np.random.uniform(low=-1, high=1, size=(indim, outdim)))
        self.biases = Tensor(np.zeros(shape=(outdim, ), dtype=np.float32))
        
        self.weights.requires_grad = True
        self.biases.requires_grad = True

    def __call__(self, x):
        x = x if isinstance(x, Tensor) else Tensor(x)
        out = x @ self.weights + self.biases
        
        if self.activation:
            out = out.relu()
        return out
    
    def __repr__(self):
        return f"Layer_{self.name},IN:{self.indim}, OUT:{self.outdim}, activation={self.activation}"

    def params(self):
        for tensor in [self.weights, self.biases]:
            yield tensor


class MLP:
    ID = 0

    def __init__(self, indim, hidden_dim, n_hidden_layers, outdim, activation):
        MLP.ID += 1
        self.layers = []
        self.layers += [Layer(indim=indim, outdim=hidden_dim, activation=activation, name="INPUT")]
        self.layers += [Layer(indim=hidden_dim, outdim=hidden_dim, activation=activation, name=f"HIDDEN_{i}") for i in range(n_hidden_layers)]
        self.layers += [Layer(indim=hidden_dim, outdim=outdim, activation=False, name="OUTPUT")]
        self.id = MLP.ID
    
    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
            
        return x

    def params(self):
        for layer in self.layers:
            for param in layer.params():
                yield param

    def __repr__(self):
        return "\n".join([str(l) for l in self.layers])
    

# input = np.random.random((4,3))
# mlp = MLP(indim=3, outdim=1, hidden_dim=16, n_hidden_layers=1, activation=True)
# print(input)
# print(mlp)
# print(mlp(input))