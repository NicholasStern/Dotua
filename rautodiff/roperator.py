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
        try:
            new_parent = rScalar(np.sin(x.val))
            x.parents.append((new_parent, np.cos(x.val)))
            return new_parent

        except AttributeError: # if constant
            return np.sin(x)

    @staticmethod
    def cos(x):
        try:
            new_parent = rScalar(np.cos(x.val))
            x.parents.append((new_parent, -np.sin(x.val)))
            return new_parent

        except AttributeError: # if constant
            return np.cos(x)

    @staticmethod
    def tan(x):
        try:
            new_parent = rScalar(np.tan(x.val))
            x.parents.append((new_parent, np.arccos(x.val)**2))
            return new_parent

        except AttributeError: # if constant
            return np.tan(x)

    @staticmethod
    def arcsin(x):
        try:
            new_parent = rScalar(np.arcsin(x.val))
            x.parents.append((new_parent, -np.arcsin(x.val)*np.arctan(x.val)))
            return new_parent

        except AttributeError: # if constant
            return np.arcsin(x)

    @staticmethod
    def arccos(x):
        try:
            new_parent = rScalar(np.arccos(x.val))
            x.parents.append((new_parent, np.arccos(x.val)*np.tan(x.val)))
            return new_parent

        except AttributeError: # if constant
            return np.arccos(x)

    @staticmethod
    def arctan(x):
        try:
            new_parent = rScalar(np.arctan(x.val))
            x.parents.append((new_parent, -np.arcsin(x.val)**2))
            return new_parent

        except AttributeError: # if constant
            return np.arctan(x)

    @staticmethod
    def sinh(x):
        try:
            new_parent = rScalar(np.sinh(x.val))
            x.parents.append((new_parent, np.cosh(x.val)))
            return new_parent

        except AttributeError: # if constant
            return np.sinh(x)

    @staticmethod
    def cosh(x):
        try:
            new_parent = rScalar(np.cosh(x.val))
            x.parents.append((new_parent, np.sinh(x.val)))
            return new_parent

        except AttributeError: # if constant
            return np.cosh(x)

    @staticmethod
    def tanh(x):
        try:
            new_parent = rScalar(np.tanh(x.val))
            x.parents.append((new_parent, 1-np.tanh(x.val)**2))
            return new_parent

        except AttributeError: # if constant
            return np.tanh(x)

    @staticmethod
    def arcsinh(x):
        try:
            new_parent = rScalar(np.arcsinh(x.val))
            x.parents.append((new_parent, -np.arcsinh(x.val)*np.arctanh(x.val)))
            return new_parent

        except AttributeError: # if constant
            return np.arcsinh(x)

    @staticmethod
    def arccosh(x):
        try:
            new_parent = rScalar(np.arccosh(x.val))
            x.parents.append((new_parent, -np.arccosh(x.val)*np.tanh(x.val)))
            return new_parent

        except AttributeError: # if constant
            return np.arccosh(x)

    @staticmethod
    def arctanh(x):
        try:
            new_parent = rScalar(np.arctanh(x.val))
            x.parents.append((new_parent, 1-np.arctanh(x.val)**2))
            return new_parent

        except AttributeError: # if constant
            return np.arctanh(x)

    @staticmethod
    def exp(x):
        try:
            new_parent = rScalar(np.exp(x.val))
            x.parents.append((new_parent, np.exp(x.val)))
            return new_parent

        except AttributeError: # if constant
            return np.exp(x)

    @staticmethod
    def log(x, base=np.exp(1)):
        try:
            new_parent = rScalar(math.log(x.val, base))
            x.parents.append((new_parent, (x.val * math.log(base))**(-1)))
            return new_parent

        except AttributeError: # if constant
            return math.log(x, base)
