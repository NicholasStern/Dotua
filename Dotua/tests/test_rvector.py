from Dotua.rautodiff import rAutoDiff as rad
from Dotua.roperator import rOperator as rop
import numpy as np

ad = rad()
op = rop()
x,y = ad.create_rvector([[1, 2, 3], [1,3,6]])

def test_eval():
    assert x.eval() == x.val
    assert y.eval() == y.val

def test_add():
	f = x + y
	g = x + 1
	h = 3 + x
	assert ad.partial(f,x) == list([1,1,1])
	assert ad.partial(g,x) == list([1,1,1])
	assert ad.partial(h,x) == list([1,1,1])

def test_subtract():
	f = x - y
	g = x - 3
	h = 3 - x
	assert ad.partial(f,x) == list([1,1,1])
	assert ad.partial(g,x) == list([1,1,1])
	assert ad.partial(h,x) == list([-1,-1,-1])

def test_mul():
	f = x * y
	g = x * 3
	h = 3 * x
	assert ad.partial(f,x) == list(y.val)
	assert ad.partial(g,x) == list([3,3,3])
	assert ad.partial(h,x) == list([3,3,3])

def test_devide():
	f = x / y
	g = x / 3
	h = 3 / x
	assert ad.partial(f,x) == list(1/y.val)
	assert ad.partial(g,x) == list([1/3,1/3,1/3])
	assert ad.partial(h,x) == list(-3/np.power(x,2))

def test_neg():
	f = -x
	assert ad.partial(f,x) == list([-1,-1,-1])

