from .node import Node


class Scalar(Node):
    def __init__(self, val, der=1, j=None, u=None):
        self._val = val
        self._der = der
        self._jacobian = j
        self._universe = u

    def init_jacobian(self, nodes):
        '''
        Call this function once the autodiff driver has initialized all of the
        users requested variables
        '''
        self._universe = nodes
        self._jacobian = {node: (id(self) == id(node)) for node in nodes}
        print(self._jacobian)

    def eval(self):
        return self._val, self._der

    def get_partial(self, var):
        return self._jacobian[var]

    def __add__(self, other):
        new_node = Scalar(self._val, self._der, self._jacobian, self._universe)
        try:
            assert(self._universe == other._universe)
            new_node._val += other._val
            new_node._der += other._der
            new_node._jacobian = {k: (v + other.get_partial(k))
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
        new_node = Scalar(self._val, self._der, self._jacobian, self._universe)
        try:
            new_node._val *= other._val
            new_node._der = self._val * other._der + self._der * other._val
            new_node._jacobian = {k: self._val * other.get_partial(k)
                                  + self.get_partial(k) * other._val
                                  for k, v in self._jacobian.items()}
        except:
            new_node._val *= other
            new_node._der *= other
            new_node._jacobian = {k: v * other
                                  for k, v in self._jacobian.items()}
        return new_node

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        raise NotImplemented

    def __rtruediv__(self, other):
        raise NotImplemented

    def __neg__(self):
        new_node = Scalar(-1 * self._val, -1 * self._der)
        new_node._jacobian = {k: -1 * v for k, v in self._jacobian.items()}
        new_node._universe = self._universe
        return new_node
