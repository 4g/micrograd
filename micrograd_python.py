import random
from typing import Any

class Value:
    ID = 0
    def __init__(self,
                 data,
                 _op=None,
                 _children=()):
        
        Value.ID += 1
        self.data = data
        self.grad = 0

        self._op = _op
        # the tree is upside down in order of compute
        # node which calls backward is root node
        # nodes at the beginning of calculation are leaves
        self._children = set(_children)
        self._backward = lambda : None
        self.id = Value.ID + 1
        
        # print(self)

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, _children=(), _op='')
        out = Value(self.data + other.data, _op='+', _children=[self, other])

        def _backward():
            # applying the chain rule
            # root = z(f(x)) 
            # d(root)/d(x) = dz * df
            # f = x + y , df/dx = 1
            self.grad += 1 * out.grad
            other.grad += 1 * out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, _children=(), _op='')
        out = Value(self.data * other.data, _op='*', _children=[self, other])
        
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out


    def __neg__(self):
        return self * -1
    
    def __radd__(self, other):
        return self + other 

    def __rmul__(self, other):
        return self * other
    
    def __rsub__(self, other):
        return self - other
    

    def __pow__(self, other):
        assert isinstance(other, int)
        out = Value(self.data ** other, _op='pow', _children=(self, ))
        def _backward():
            self.grad += other * (self.data ** (other-1)) * out.grad
        
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(self.data, 0), _op='relu', _children=(self,))
        def _backward():
            self.grad += 0 if self.data < 0 else out.grad
        
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)


    def backward(self):
        self.grad = 1.0
        topo = []
        visited = set()

        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        # print(topo)
        for v in reversed(topo):
            v._backward()

    def __repr__(self) -> str:
        return f"Node(id: {self.id}, data:{self.data}, grad:{self.grad}, op:{self._op})"


class Neuron:
    ID = 0

    def __init__(self, indim, activation=True):
        Neuron.ID += 1
        self.indim = indim
        self.w = [Value(random.uniform(-1,1), _op='w') for i in range(indim)]
        self.b = Value(0, _op='b')
        self.id = Neuron.ID
        self.activation = activation

    def __call__(self, x):
        value = sum(_x*_w for _x, _w in zip(x, self.w)) + self.b
        if self.activation:
            value = value.relu()
        
        return value

    def __repr__(self):
        return f"Neu:{self.id},IN:{self.indim},act={self.activation}"

    def params(self):
        for param in self.w + [self.b]:
            yield param


class Layer:
    ID = 0

    def __init__(self, indim, outdim, activation=True):
        Layer.ID += 1
        self.indim = indim
        self.outdim = outdim
        self.activation = activation
        self.neurons = [Neuron(indim=indim, activation=activation) for i in range(outdim)]
        self.id = Layer.ID

    def __call__(self, x):
        out = [neuron(x) for neuron in self.neurons]
        return out
    
    def __repr__(self):
        return f"Layer:{self.id},IN:{self.indim}, OUT:{self.outdim}, activation={self.activation}"

    def params(self):
        for neuron in self.neurons:
            for param in neuron.params():
                yield param


class MLP:
    ID = 0

    def __init__(self, indim, hidden_dim, n_hidden_layers, outdim, activation):
        MLP.ID += 1
        self.layers = []
        self.layers += [Layer(indim=indim, outdim=hidden_dim, activation=activation)]
        self.layers += [Layer(indim=hidden_dim, outdim=hidden_dim, activation=activation) for i in range(n_hidden_layers)]
        self.layers += [Layer(indim=hidden_dim, outdim=outdim, activation=False)]
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
