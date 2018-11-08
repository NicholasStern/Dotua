# This file serves to test the operator.py module
from ..operator import Operator as op
from ..autodiff import AutoDiff as ad
import numpy as np

x, z = tuple(ad.create_scalar([0, 1]))
y = 0


def test_sin():
    # Autodiff Obj
    sx = op.sin(x)
    assert np.sin(x._val) == sx._val
    for k in x._jacobian.keys():
        assert(sx.partial(k) == x.partial(k) * np.cos(x._val))
    # Constant
    assert op.sin(y) == np.sin(y)


def test_cos():
    # Autodiff Obj
    cx = op.cos(x)
    assert np.cos(x._val) == cx._val
    for k in x._jacobian.keys():
        assert(cx.partial(k) == -x.partial(k) * np.sin(x._val))
    # Constant
    assert op.cos(y) == np.cos(y)


def test_tan():
    # Autodiff Obj
    tx = op.tan(x)
    assert np.tan(x._val) == tx._val
    for k in x._jacobian.keys():
        assert(tx.partial(k) == x.partial(k) * np.arccos(x._val)**2)

    # Constant
    assert op.tan(y) == np.tan(y)


def test_arcsin():
    # Autodiff Obj
    asx = op.arcsin(x)
    assert np.arcsin(x._val) == asx._val
    for k in x._jacobian.keys():
        assert(asx.partial(k) == -x.partial(k) * np.arcsin(x._val)
               * np.arctan(x._val))
    # Constant
    assert op.arcsin(y) == np.arcsin(y)


def test_arccos():
    # Autodiff Obj
    acx = op.arccos(x)
    assert np.arccos(x._val) == acx._val
    for k in x._jacobian.keys():
        assert(acx.partial(k) == x.partial(k) * np.arccos(x._val)
               * np.tan(x._val))

    # Constant
    assert op.arccos(y) == np.arccos(y)


def test_arctan():
    # Autodiff Obj
    atx = op.arctan(x)
    assert np.arctan(x._val) == atx._val
    for k in x._jacobian.keys():
        assert(atx.partial(k) == -x.partial(k) * np.arcsin(x._val)**2)

    # Constant
    assert op.arctan(y) == np.arctan(y)


def test_sinh():
    # Autodiff Obj
    shx = op.sinh(x)
    assert np.sinh(x._val) == shx._val
    for k in x._jacobian.keys():
        assert(shx.partial(k) == x.partial(k) * np.cosh(x._val))

    # Constant
    assert op.sinh(y) == np.sinh(y)


def test_cosh():
    # Autodiff Obj
    chx = op.cosh(x)
    assert np.cosh(x._val) == chx._val
    for k in x._jacobian.keys():
        assert(chx.partial(k) == x.partial(k) * np.sinh(x._val))

    # Constant
    assert op.cosh(y) == np.cosh(y)


def test_tanh():
    # Autodiff Obj
    thx = op.tanh(x)
    assert np.tanh(x._val) == thx._val
    for k in x._jacobian.keys():
        assert(thx.partial(k) == x.partial(k) * (1 - np.tanh(x._val)**2))

    # Constant
    assert op.tanh(y) == np.tanh(y)


def test_arcsinh():
    # Autodiff Obj
    ashx = op.arcsinh(x)
    assert np.arcsinh(x._val) == ashx._val
    for k in x._jacobian.keys():
        assert(ashx.partial(k) == -x.partial(k) * np.arcsinh(x._val)
               * np.arctanh(x._val))

    # Constant
    assert op.arcsinh(y) == np.arcsinh(y)


def test_arccosh():
    x = ad.create_scalar(1)
    y = 2
    # Autodiff Obj
    achx = op.arccosh(x)
    assert np.arccosh(x._val) == achx._val
    for k in x._jacobian.keys():
        assert(achx.partial(k) == -x.partial(k) * np.arccosh(x._val)
               * np.tanh(x._val))

    # Constant
    assert op.arccosh(y) == np.arccosh(y)


def test_arctanh():
    # Autodiff Obj
    athx = op.arctanh(x)
    assert np.arctanh(x._val) == athx._val
    for k in x._jacobian.keys():
        assert(athx.partial(k) == x.partial(k) * (1 - np.arctanh(x._val)**2))

    # Constant
    assert op.arctanh(y) == np.arctanh(y)

def test_exp():
    # Autodiff Obj
    ex = op.exp(x)
    assert np.exp(x._val) == ex._val
    for k in x._jacobian.keys():
        assert(ex.partial(k) == x.partial(k) * np.exp(x._val))

    # Constant
    assert op.exp(y) == np.exp(y)

def test_add():
    res = op.sin(x) + op.sin(z)
    print('res: ', res)
    print('res._val: ', res._val)
    print(np.sin(x._val) + np.sin(z._val))

    assert np.sin(x._val) + np.sin(z._val) == res._val
