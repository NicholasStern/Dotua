from .rnode import rNode
import numpy as np

class rScalar(rNode):
    def __init__(self, val):
        self.val = val
        self.parents = []
        self.grad_val = None


    def gradient(self):
        if self.grad_val is None:
            self.grad_val = 0
            for par, val in self.parents:
                self.grad_val += par.gradient() * val
        return self.grad_val

    def __add__(self, other):
        new_parent = Scalar(self.val + other.val)
        self.parents.append((new_parent, 1))
        other.parents.append((new_parent, 1))
        return new_parent

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        new_parent = Scalar(self.val * other.val)
        self.parents.append((new_parent, other.val))
        other.parents.append((new_parent, self.val))
        return new_parent

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        '''
        Need to calculate @self / @other
        '''
        pass

    def __rtruediv__(self, other):
        '''
        Need to calculate @other / @self
        '''
        pass

    def __pow__(self, other):
        '''
        Need to calculate @self ** @other
        '''
        pass

    def __rpow__(self, other):
        '''
        Need to calculate @other ** @self
        '''
        pass

    def __neg__(self):
        pass
