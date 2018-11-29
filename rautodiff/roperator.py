import math
import numpy as np
from rautodiff.rnodes.rscalar import rScalar


class rOperator:
    """Returns a new scalar object subject to the operator and propagates the
    value and derivative according to the chain rule

    The example below pertains to an action on an autodiff scalar: x

    Example Usage
    -------------

        $ import autodiff as ad
        $ x = ad.create_scalar(0)
        $ from autodiff.operators import Operator as op
        $ y = op.sin(x)

    """
    @staticmethod
    def sin(x):
        # implement try/except
        new_parent = rScalar(np.sin(x.val))
        x.parents.append((new_parent, np.cos(x.val)))
        return new_parent

    @staticmethod
    def cos(x):
        pass


    @staticmethod
    def tan(x):
        pass

    @staticmethod
    def arcsin(x):
        pass


    @staticmethod
    def arccos(x):
        pass

    @staticmethod
    def arctan(x):
        pass

    @staticmethod
    def sinh(x):
        pass

    @staticmethod
    def cosh(x):
        pass

    @staticmethod
    def tanh(x):
        pass

    @staticmethod
    def arcsinh(x):
        pass

    @staticmethod
    def arccosh(x):
        pass

    @staticmethod
    def arctanh(x):
        pass

    @staticmethod
    def exp(x):
        pass

    @staticmethod
    def log(x, base=np.exp(1)):
        pass
