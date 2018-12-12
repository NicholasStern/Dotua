from Dotua.rautodiff import rAutoDiff as rad
from Dotua.roperator import rOperator as rop
import numpy as np

ad = rad()
op = rop()
x,y = ad.create_rvector([[1, 2, 3], [1,3,6]])

def test_eval():
    assert list(x.eval()) == list(x.val)
    assert list(y.eval()) == list(y.val)

def test_add():
	f = x + y
	g = x + 1
	h = 3 + x
	assert list(ad.partial(f,x)) == list([1,1,1])
	assert list(ad.partial(g,x)) == list([1,1,1])
	assert list(ad.partial(h,x)) == list([1,1,1])

def test_subtract():
	f = x - y
	g = x - 3
	h = 3 - x
	assert list(ad.partial(f,x)) == list([1,1,1])
	assert list(ad.partial(g,x)) == list([1,1,1])
	assert list(ad.partial(h,x)) == list([-1,-1,-1])

def test_mul():
	f = x * y
	g = x * 3
	h = 3 * x
	assert list(ad.partial(f,x)) == list(y.val)
	assert list(ad.partial(g,x)) == list([3,3,3])
	assert list(ad.partial(h,x)) == list([3,3,3])

def test_devide():
	f = x / y
	g = x / 3
	h = 3 / x
	assert list(ad.partial(f,x)) == list(1/y.val)
	assert list(ad.partial(g,x)) == list([1/3,1/3,1/3])
	#print(ad.partial(f,x))
	#print(ad.partial(g,x))

def test_neg():
	f = -x
	print(ad.partial(f,x))
	assert list(ad.partial(f,x)) == list([-1,-1,-1])

def test_pow():
	f = x ** y
	g = x ** 2
	h = 2 ** x
	#print(ad.partial(g,x))
	assert list(ad.partial(f,x)) == list(y.val*x.val**(y.val-1))
	assert list(ad.partial(g,x)) == list(2 * x.val)

test_devide()
test_neg()
test_pow()

