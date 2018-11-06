'''
This class will serve as an abstract interface for the @Scalar and @Vector
classes.
'''


class Node():
    def __init__(self):
        pass

    def __add__(self, other):
        raise NotImplemented

    def __radd__(self, other):
        raise NotImplemented

    def __sub__(self, other):
        raise NotImplemented

    def __rsub__(self, other):
        raise NotImplemented

    def __mul__(self, other):
        raise NotImplemented

    def __rmul__(self, other):
        raise NotImplemented

    def __truediv__(self, other):
        raise NotImplemented

    def __rtruediv__(self, other):
        raise NotImplemented

    def __neg__(self):
        raise NotImplemented
