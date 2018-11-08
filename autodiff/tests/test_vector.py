import numpy as np
from ..nodes.vector import Vector

'''
Test for vector variable basic functions and the jacobian of vector function to vector
'''

# Define vector objects
x = Vector([1,2])
y = Vector([0,1])

f_1 = x[0] + x[1]
f_2 = x[0] - 3 - x[1] - x[1]
f_3 = 1 + x[1] - x[0] * x[1] - x[0] - 3
f_4 = 3 / x[0] + (x[0] * x[1]) / 2 - 4 * x[1]
f_5 = x[0] ** 3 + x[1] / x[0]
f_6 =  2 ** x[1]
f_7 = 3 + x + y - 2
f_8 = -1 - x - y - 3
f_9 = 3 * x + y * 2 + x * y
f_10 = y / 3 + 2 / x + y / x

def test_add():
	assert(f_1.eval() == (3, [1,1]))

def test_sub():
	assert(f_2.eval() == (-6, [1,-2]))

def test_mul():
	assert(f_3.eval() == (-3, [-3,0]))

def test_devide():
	assert(f_4.eval() == (-4, [-2,-3.5]))

def test_pow():
	assert(f_5.eval() == (3, [1,1]))

def test_rpow():
	assert(f_6.eval() == (4, [0,4*np.log(2)]))

def test_vector_add():
	assert(f_7.eval() == [2,4])

def test_vector_sub():
	assert(f_8.eval() == [-5,-7])

def test_vector_times():
	assert(f_9.eval() == [3,10])

def test_vector_divide():
	assert(f_10.eval() == [2,11/6])

# Define a vector function and get a jacobian of the vector function to vector

f = [f_1, f_2, f_3, f_4, f_5]
jacobian = []

for function in f:
	jacobian.append(function.eval()[1])

def test_jacobian():
	assert(jacobian == [[1,1], [1,-2], [-3,0], [-2,-3.5], [1,1]])

