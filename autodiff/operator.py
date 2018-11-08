import numpy as np
from autodiff.nodes.scalar import Scalar

class Operator:

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
    def __init__(self):
        pass

    @staticmethod
    def sin(x):
        try:
            jacobian = {k: x.partial(k) * np.cos(x._val)
                        for k in x._jacobian.keys()}
            return Scalar(np.sin(x._val), jacobian)
        except AttributeError:
            return np.sin(x)

    @staticmethod
    def cos(x):
        try:
            jacobian = {k: x.partial(k) * -np.sin(x._val)
                        for k in x._jacobian.keys()}
            return Scalar(np.cos(x._val), jacobian)
        except AttributeError:
            return np.cos(x)

    @staticmethod
    def tan(x):
        try:
            jacobian = {k: x.partial(k) * np.arccos(x._val)**2
                        for k in x._jacobian.keys()}
            return Scalar(np.tan(x._val), jacobian)
        except AttributeError:
            return np.tan(x)

    @staticmethod
    def arcsin(x):
        try:
            jacobian = {k: x.partial(k) * -np.arcsin(x._val)*np.arctan(
                x._val) for k in x._jacobian.keys()}
            return Scalar(np.arcsin(x._val), jacobian)
        except AttributeError:
            return np.arcsin(x)

    @staticmethod
    def arccos(x):
        try:
            jacobian = {k: x.partial(k) * np.arccos(x._val)*np.tan(
                x._val) for k in x._jacobian.keys()}
            return Scalar(np.arccos(x._val), jacobian)
        except AttributeError:
            return np.arccos(x)

    @staticmethod
    def arctan(x):
        try:
            jacobian = {k: x.partial(k) * -np.arcsin(x._val)**2
                        for k in x._jacobian.keys()}
            return Scalar(np.arctan(x._val), jacobian)
        except AttributeError:
            return np.arctan(x)

    @staticmethod
    def sinh(x):
        try:
            jacobian = {k: x.partial(k) * np.cosh(x._val)
                        for k in x._jacobian.keys()}
            return Scalar(np.sinh(x._val), jacobian)
        except AttributeError:
            return np.sinh(x)

    @staticmethod
    def cosh(x):
        try:
            jacobian = {k: x.partial(k) * np.sinh(x._val)
                        for k in x._jacobian.keys()}
            return Scalar(np.cosh(x._val), jacobian)
        except AttributeError:
            return np.cosh(x)

    @staticmethod
    def tanh(x):
        try:
            jacobian = {k: x.partial(k) * (1-np.tanh(x._val)**2)
                        for k in x._jacobian.keys()}
            return Scalar(np.tanh(x._val), jacobian)
        except AttributeError:
            return np.tanh(x)

    @staticmethod
    def arcsinh(x):
        try:
            jacobian = {k: x.partial(k) * -np.arcsinh(x._val)*np.arctanh(
                x._val) for k in x._jacobian.keys()}
            return Scalar(np.arcsinh(x._val), jacobian)
        except AttributeError:
            return np.arcsinh(x)

    @staticmethod
    def arccosh(x):
        try:
            jacobian = {k: x.partial(k) * -np.arccosh(x._val)*np.tanh(
                x._val) for k in x._jacobian.keys()}
            return Scalar(np.arccosh(x._val), jacobian)
        except AttributeError:
            return np.arccosh(x)

    @staticmethod
    def arctanh(x):
        try:
            jacobian = {k: x.partial(k) * (1-np.arctanh(x._val)**2)
                        for k in x._jacobian.keys()}
            return Scalar(np.arctanh(x._val), jacobian)
        except AttributeError:
            return np.arctanh(x)

    @staticmethod
    def exp(x):
        try:
            jacobian = {k: x.partial(k) * np.exp(x._val)
                        for k in x._jacobian.keys()}
            return Scalar(np.exp(x._val), jacobian)
        except AttributeError:
            return np.exp(x)
