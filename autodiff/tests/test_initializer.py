'''
This file tests that the AutoDiff initializer functions defined in autodiff.py
are correct.
'''
from ..autodiff import AutoDiff


# Test creating a single variable
def test_single_variable():
    x = AutoDiff.create_scalar(5)
    assert(x.eval() == (5, {x: 1}))


# Test initializing multiple variables at once
def test_multiple_variables():
    x, y, z = tuple(AutoDiff.create_scalar([1, 2, 3]))
    assert(x.eval() == (1, {x: 1, y: 0, z: 0}))
    assert(y.eval() == (2, {x: 0, y: 1, z: 0}))
    assert(z.eval() == (3, {x: 0, y: 0, z: 1}))

# Test creating a single vector
def test_single_vector():
	x = AutoDiff.create_vector([5])
	assert x._val == 5

# Test creating multiple vectors
def test_multiple_vectors():
	x, y = AutoDiff.create_vector([1, 2])
	assert x._val == 1
	assert y._val == 2
