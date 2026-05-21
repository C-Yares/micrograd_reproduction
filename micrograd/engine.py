
class Value:
    def __init__(self, data, _prev = set(), _op = ''):
        self.data = data
        self.grad = 0.0
        self._prev = set(_prev)
        self._op = _op
        self._backward = lambda: None
    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, {self, other}, '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out
    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, {self, other}, '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out
    def backward(self):
    # Build topological order of all nodes reachable from self
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)
        build_topo(self)

        # Set gradient at the output (root of backward pass) to 1
        self.grad = 1.0

        # Walk in reverse topological order, calling each _backward
        for node in reversed(topo):
            node._backward()
    def __pow__(self, n):
        assert isinstance(n, (int, float)), "only supporting scalar exponents"
        out = Value(self.data ** n, _prev=set({self}), _op=f"**{n}")
        def _backward():
            self.grad += (n*self.data ** (n-1)) * out.grad
        out._backward = _backward
        return out
    def __radd__(self, other):
        return self + other
    def __sub__(self, other):
        return self + (-other)
    def __rsub__(self, other):
        return other + (-self)
    def __neg__(self):
        return self * -1
    def __rmul__(self, other): # other * self
        return self * other
    def __truediv__(self, other): # self / other
        return self * other**-1
    def __rtruediv__(self, other): # other / self
        return other * self**-1

    def relu(self):
        out = Value(max(0, self.data), _prev={self}, _op='relu')
        def _backward():
            self.grad += (out.data > 0) * out.grad   # boolean → 0 or 1
            # equivalently: (self.data > 0) * out.grad
        out._backward = _backward
        return out
        
    
