import numpy as np

class Value:
    def __init__(self,
                 data,
                 _op,
                 _children,
                 store_type=np.float32,
                 compute_type=np.float32):

        self.data = store_type(data)
        self.grad = 0

        self._op = _op
        # the tree is upside down in order of compute
        # node which calls backward is root node
        # nodes at the beginning of calculation are leaves
        self._children = set(_children)
        self._backward = None

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, _children=None, _op='')
        out = Value(self.data + other.data, _op='+', _children=[self, other])

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __sub__(self, other):

    def __mul__(self, other):

    def __neg__(self, other):

    def backward(self, grad=1.0):
        self.grad = grad

        for prev in self.prev:
            prev.backward(grad)
