from .node import Node


class Scalar(Node):
    def __init__(self, val, der=None):
        self._val = val
        self._jacobian = der

    def init_jacobian(self, nodes):
        '''
        Call this function once the autodiff driver has initialized all of the
        users requested variables
        '''
        self._jacobian = {node: int(id(self) == id(node)) for node in nodes}
        print(self._jacobian)

    def eval(self):
        return self._val, self._jacobian

    def partial(self, var):
        return self._jacobian[var]

    def __add__(self, other):
        new_node = Scalar(self._val, self._jacobian)
        try:
            new_node._val += other._val
            new_node._jacobian = {k: (v + other.partial(k))
                                  for k, v in self._jacobian.items()}
        except AttributeError:
            new_node._val += other
        return new_node

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        new_node = Scalar(self._val, self._jacobian)
        try:
            new_node._val *= other._val
            new_node._jacobian = {k: self._val * other.partial(k)
                                  + v * other._val
                                  for k, v in self._jacobian.items()}
        except:
            new_node._val *= other
            new_node._jacobian = \
                {k: v * other for k, v in self._jacobian.items()}
        return new_node

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        '''
        Need to calculate @self / @other
        '''
        new_node = Scalar(self._val, self._jacobian)
        try:
            new_node._val /= other._val
            new_node._jacobian = \
                {k: (v * other._val - self._val * other.partial(k))
                 / (other._val ** 2) for k, v in self._jacobian.items()}
        except AttributeError:
            new_node._val /= other
            new_node._jacobian = \
                {k: v / other for k, v in self._jacobian.items()}
        return new_node

    def __rtruediv__(self, other):
        '''
        Need to calculate @other / @self
        '''
        new_node = Scalar(self._val, self._jacobian)
        try:
            new_node._val = other._val / self._val
            new_node._jacobian = \
                {k: (self._val * other.partial(k) - v * other._val)
                 / (self._val ** 2)for k, v in self._jacobian.items()}
        except AttributeError:
            new_node._val = other / self._val
            new_node._jacobian = \
                {k: other * (-v) / (self._val ** 2)
                 for k, v in self._jacobian.items()}
        return new_node

    def __neg__(self):
        jacobian = {k: -1 * v for k, v in self._jacobian.items()}
        return Scalar(-1 * self._val, jacobian)
