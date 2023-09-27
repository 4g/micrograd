from torch import Tensor, tensor
import numpy as np
import torch

class Layer:
    ID = 0

    def __init__(self, indim, outdim, activation=True, name=None, device='cpu'):
        Layer.ID += 1

        self.id = Layer.ID
        self.name = name if name else f"LAYER_{Layer.ID}"
        self.indim = indim
        self.outdim = outdim
        self.activation = activation
        self.device = device
        self.weights = tensor(np.random.uniform(low=-1, high=1, size=(indim, outdim)), device=device, dtype=torch.float32)
        self.biases = tensor(np.zeros(shape=(outdim, ), dtype=np.float32), device=device, dtype=torch.float32)
        
        self.weights.requires_grad = True
        self.biases.requires_grad = True

    def __call__(self, x):
        x = x if isinstance(x, Tensor) else tensor(x, device=self.device, dtype=torch.float32)
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

    def __init__(self, indim, hidden_dim, n_hidden_layers, outdim, activation, device):
        MLP.ID += 1
        self.layers = []
        self.layers += [Layer(indim=indim,
                              outdim=hidden_dim,
                              activation=activation,
                              name="INPUT",
                              device=device)]

        self.layers += [Layer(indim=hidden_dim,
                              outdim=hidden_dim,
                              activation=activation,
                              name=f"HIDDEN_{i}",
                              device=device) for i in range(n_hidden_layers)]

        self.layers += [Layer(indim=hidden_dim,
                              outdim=outdim,
                              activation=False,
                              name="OUTPUT",
                              device=device)]
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

    def save(self, path):
        for idx, param in enumerate(self.params()):
            torch.save(param, f'{path}/{idx}.tensor')

    def load(self, path):
        for idx, param in enumerate(self.params()):
            p = torch.load(f'{path}/{idx}.tensor')
            param.data = p.data
