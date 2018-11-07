# This file serves to test the operator.py module
from ..operators.operator import Operator as op
from ..nodes.scalar import Scalar
import numpy as np

x = Scalar(0, der=2)  # Autodiff obj w/ derivative of 2
y = 0

def test_sin():
    # Autodiff Obj
    assert np.sin(x._val) == op.sin(x)._val and x._der*np.cos(x._val) \
           == op.sin(x)._der
    # Constant
    assert op.sin(y) == np.sin(y)

def test_cos():
    # Autodiff Obj
    assert np.cos(x._val) == op.cos(x)._val and -x._der*np.sin(x._val) \
           == op.cos(x)._der
    # Constant
    assert op.cos(y) == np.cos(y)

def test_tan():
    # Autodiff Obj
    assert np.tan(x._val) == op.tan(x)._val and x._der*np.arccos(x._val)**2 \
           == op.tan(x)._der
    # Constant
    assert op.tan(y) == np.tan(y)

def test_arcsin():
    # Autodiff Obj
    assert np.arcsin(x._val) == op.arcsin(x)._val and -x._der*np.arcsin(x._val)\
           *np.arctan(x._val) == op.arcsin(x)._der
    # Constant
    assert op.arcsin(y) == np.arcsin(y)

def test_arccos():
    # Autodiff Obj
    assert np.arccos(x._val) == op.arccos(x)._val and x._der*np.arccos(x._val)\
           *np.tan(x._val) == op.arccos(x)._der
    # Constant
    assert op.arccos(y) == np.arccos(y)

def test_arctan():
    # Autodiff Obj
    assert np.arctan(x._val) == op.arctan(x)._val and -x._der*np.arcsin(x._val)**2 \
           == op.arctan(x)._der
    # Constant
    assert op.arctan(y) == np.arctan(y)

def test_sinh():
    # Autodiff Obj
    assert np.sinh(x._val) == op.sinh(x)._val and x._der*np.cosh(x._val) \
           == op.sinh(x)._der
    # Constant
    assert op.sinh(y) == np.sinh(y)

def test_cosh():
    # Autodiff Obj
    assert np.cosh(x._val) == op.cosh(x)._val and x._der*np.sinh(x._val) \
           == op.cosh(x)._der
    # Constant
    assert op.cosh(y) == np.cosh(y)

def test_tanh():
    # Autodiff Obj
    assert np.tanh(x._val) == op.tanh(x)._val and x._der*(1-np.tanh(x._val)**2) \
           == op.tanh(x)._der
    # Constant
    assert op.tanh(y) == np.tanh(y)

def test_arcsinh():
    # Autodiff Obj
    assert np.arcsinh(x._val) == op.arcsinh(x)._val and -x._der*np.arcsinh(x._val)\
           *np.arctanh(x._val) == op.arcsinh(x)._der
    # Constant
    assert op.arcsinh(y) == np.arcsinh(y)

def test_arccosh():
    # Autodiff Obj
    assert np.arccosh(x._val) == op.arccosh(x)._val and -np.arccosh(x._val)\
           *np.tanh(x._val) == op.arccosh(x)._der
    # Constant
    assert op.arccosh(y) == np.arccosh(y)

def test_arctanh():
    # Autodiff Obj
    assert np.arctanh(x._val) == op.arctanh(x)._val and x._der*\
           (1-np.arctanh(x._val)**2) == op.arctanh(x)._der
    # Constant
    assert op.arctanh(y) == np.arctanh(y)

def test_exp():
    # Autodiff Obj
    assert np.exp(x._val) == op.exp(x)._val and x._der*np.exp(x._val) == op.exp(x)._der
    # Constant
    assert op.exp(y) == np.exp(y)
