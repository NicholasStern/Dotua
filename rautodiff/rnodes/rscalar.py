import numpy as np


class rScalar():
    def __init__(self, val):
        self.val = val
        self.parents = []
        self.grad_val = None

    def gradient(self):
        if self.grad_val is None:
            self.grad_val = 0
            for parent, val in self.parents:
                self.grad_val += parent.gradient() * val
        return self.grad_val

    def __add__(self, other):
        new_parent = rScalar(self.val)
        try:
            new_parent.val += other.val
            self.parents.append((new_parent, 1))
            other.parents.append((new_parent, 1))
        except AttributeError:
            new_parent.val += other
            self.parents.append((new_parent, 1))
        return new_parent

    def __radd__(self, other):
        return self + other

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __mul__(self, other):
        new_parent = rScalar(self.val)
        try:
            new_parent.val *= other.val
            self.parents.append((new_parent, other.val))
            other.parents.append((new_parent, self.val))
        except AttributeError:
            new_parent.val *= other
            self.parents.append((new_parent, other))
        return new_parent

    def __rmul__(self, other):
        return self * other

    def __truediv__(self, other):
        '''
        Need to calculate @self / @other
        '''
        new_parent = rScalar(self.val)
        try:
            new_parent.val /= other.val
            self.parents.append((new_parent, 1 / other.val))
            other.parents.append((new_parent, -self.val / (other.val ** 2)))
        except AttributeError:
            new_parent.val /= other
            self.parents.append((new_parent, 1 / other))
        return new_parent

    def __rtruediv__(self, other):
        '''
        Need to calculate @other / @self
        '''
        new_parent = rScalar(self.val)
        new_parent.val = other / new_parent.val
        self.parents.append((new_parent, -other / (self.val ** 2)))
        return new_parent

    def __pow__(self, other):
        '''
        Need to calculate @self ** @other
        '''
        new_parent = rScalar(self.val)
        try:
            new_parent.val **= other.val
            self.parents.append((new_parent,
                                other.val * self.val ** (other.val - 1)))
            other.parents.append((new_parent,
                                 self.val ** other.val * np.log(self.val)))
        except AttributeError:
            new_parent.val **= other
            self.parents.append((new_parent, other * self.val ** (other - 1)))
        return new_parent

    def __rpow__(self, other):
        '''
        Need to calculate @other ** @self
        '''
        new_parent = rScalar(self.val)
        new_parent.val = other ** self.val
        self.parents.append((new_parent, other ** self.val * np.log(other)))
        return new_parent

    def __neg__(self):
        new_parent = rScalar(-self.val)
        self.parents.append((new_parent, -1))
        return new_parent
