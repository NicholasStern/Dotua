import numpy as np
from scalar import Scalar

class Operator:

    """Returns a new scalar object subject to the operator and propagates the
    value and derivative according to the chain rule

    The example below pertains to an action on an autodiff scalar: x

    Example Usage
    -------------

        >>> from operator import Operator as op
        >>> y = op.sin(x)

    """
    def __init__(self):
        pass

    @staticmethod
    def sin(x):
        return Scalar(np.sin(x.val), np.cos(x.val)*x.der)

    @staticmethod
    def cos(x):
        return Scalar(np.cos(x.val), -np.sin(x.val)*x.der)

    @staticmethod
    def tan(x):
        return Scalar(np.tan(x.val), np.arccos(x.val)**2*x.der)

    @staticmethod
    def arcsin(x):
        return Scalar(np.arcsin(x.val), -np.arcsin(x.val)*np.arctan(x.val)*x.der)

    @staticmethod
    def arccos(x):
        return Scalar(np.arccos(x.val), np.arccos(x.val)*np.tan(x.val)*x.der)

    @staticmethod
    def arctan(x):
        return Scalar(np.arctan(x.val), -np.arcsin(x.val)**2*x.der)

    @staticmethod
    def sinh(x):
        return Scalar(np.sinh(x.val), np.cosh(x.val)*x.der)

    @staticmethod
    def cosh(x):
        return Scalar(np.cosh(x.val), np.sinh(x.val)*x.der)

    @staticmethod
    def tanh(x):
        return Scalar(np.tanh(x.val), (1-np.tanh(x.val)**2)*x.der)

    @staticmethod
    def arcsinh(x):
        return Scalar(np.arcsinh(x.val), -np.arcsinh(x.val)*np.arctanh(x.val)*x.der)

    @staticmethod
    def arccosh(x):
        return Scalar(np.arccosh(x.val), -np.arccosh(x.val)*np.tanh(x.val)*x.der)

    @staticmethod
    def arctanh(x):
        return Scalar(np.arctanh(x.val), (1-np.arctanh(x.val)**2)*x.der)

