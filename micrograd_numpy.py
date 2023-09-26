import numpy as np

class Tensor:
    ID = 0
    def __init__(self,
                value=None,
                _op=None,
                _children=(),
                requires_grad=True):
        
        Tensor.ID += 1
        self.id = Tensor.ID
        
        self.data = np.asarray(value, dtype=np.float32)
        self.shape = self.data.shape
        self.requires_grad = requires_grad

        self.grad = None
        if self.requires_grad:
            self.grad = np.zeros(self.shape, dtype=np.float32)
        
        self._op = _op
        self._children = _children
        self._backward = lambda : None

    def __add__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(value=other, _children=(), _op='', requires_grad=False)
        out = Tensor(value=self.data + other.data, _op='+', _children=[self, other])


        def _backward_flat():
            if self.requires_grad:
                self.grad += out.grad
            
            if other.requires_grad:
                flatgrad = np.sum(out.grad.data, axis=0, keepdims=False)
                other.grad += flatgrad

        def _backward():
            if self.requires_grad:
                self.grad += out.grad
            
            if other.requires_grad:
                other.grad += out.grad

        if other.data.shape == self.data.shape:
            out._backward = _backward
        else:
            out._backward = _backward_flat

        return out

    def __sub__(self, other):
        return self + (-other)

    def __mul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(value=other, _children=(), _op='', requires_grad=False)
        out = Tensor(value=self.data * other.data, _op='*', _children=[self, other])

        def _backward():
            # print(self.data, other.data, out.data)
            if self.requires_grad:
                self.grad += other.data * out.grad
            
            if other.requires_grad:
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
        return other + (-self)
    
    def __pow__(self, other):
        assert isinstance(other, int)
        out = Tensor(self.data ** other, _op='pow', _children=(self, ))
        def _backward():
            self.grad += other * (self.data ** (other-1)) * out.grad
        
        out._backward = _backward
        return out

    def relu(self):
        out = Tensor(np.maximum(np.zeros(shape=self.data.shape), self.data), _op='relu', _children=(self,), requires_grad=True)
        def _backward():
            valid_grad = np.where(self.data > 0, 1.0, 0.0)
            self.grad += valid_grad * out.grad
        
        out._backward = _backward
        return out


    def __matmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(value=other, _children=(), _op='', requires_grad=False)
        out = Tensor(value=self.data @ other.data, _op='@', _children=(self, other))

        def _backward():
            if self.requires_grad:
                self.grad += np.matmul(out.grad, other.data.T)
            
            if other.requires_grad:
                other.grad += np.matmul(self.data.T, out.grad)
        
        out._backward = _backward
        return out

    def __rmatmul__(self, other):
        other = other if isinstance(other, Tensor) else Tensor(value=other, _children=(), _op='', requires_grad=False)
        return other @ self

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    def norm(self):
        return np.linalg.norm(self.data, ord=2)

    def mean(self):
        out = Tensor(value=np.mean(self.data, axis=0), _op='mean', _children=(self,), requires_grad=True)
        def _backward():
            self.grad += out.grad / np.product(self.data.shape) 
        
        out._backward = _backward
        return out

    def backward(self):
        self.grad = np.ones(shape=self.data.shape)
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
        for v in reversed(topo):
            if v.requires_grad:
                # print(v)
                v._backward()

    def __repr__(self) -> str:
        return f"** Node(id: {self.id},op:{self._op}, requires_grad:{self.requires_grad}, \n\
data{self.data.shape}:{self.data},\n\
grad{self.grad.shape}:{self.grad})\n**"


class Layer:
    ID = 0

    def __init__(self, indim, outdim, activation=True):
        Layer.ID += 1
        self.indim = indim
        self.outdim = outdim
        self.activation = activation
        self.weights = Tensor(value=np.random.random(size=(indim, outdim)))
        self.biases = Tensor(value=np.zeros(shape=(outdim, ), dtype=np.float32))
        self.id = Layer.ID

    def __call__(self, x):
        out = self.weights.__rmatmul__(x) + self.biases
        
        if self.activation:
            out = out.relu()
        return out
    
    def __repr__(self):
        return f"Layer:{self.id},IN:{self.indim}, OUT:{self.outdim}, activation={self.activation}"

    def params(self):
        for tensor in [self.weights, self.biases]:
            yield tensor


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


# x = np.random.random((3, 2))
# y = np.random.random((2, 4))
# # y = Tensor(value=y)
# # y @ x
# t1 = MLP(indim=2, outdim=1, hidden_dim=16, n_hidden_layers=2, activation=True)
# y = t1(x)

# # print(y)

# y.backward()