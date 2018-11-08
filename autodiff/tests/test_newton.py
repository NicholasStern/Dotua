from ..operators.operator import Operator as op
from ..autodiff import AutoDiff as ad
from ..examples import newton_demo as newton
import numpy as np
#from scipy import optimize
from math import isclose

def func(x):
    return op.sin(ad.create_scalar(x)[0])

#def f(x):
#    return np.sin(x)

def test_newton_method():
    assert isclose(newton.NewtonsMethod(func, 0), 0, abs_tol=1e-10)
    #optimize.newton(f,0)
