from ..nodes.vector import Vector

'''
Test for vector variable basic functions and the jacobian of vector function to vector
'''

# Define vector objects
x = Vector([1,2])

f_1 = x[0] + x[1]
f_2 = x[0] - 3 - x[1] - x[1]
f_3 = 1 + x[1] - x[0] * x[1] - x[0] - 3
f_4 = 3 / x[0] + (x[0] * x[1]) / 2 - 4 * x[1]
f_5 = x[0] ** 3 + x[1] / x[0]

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

# Define a vector function and get a jacobian of the vector function to vector

f = [f_1, f_2, f_3, f_4, f_5]
jacobian = []

for function in f:
	jacobian.append(function.eval()[1])

def test_jacobian():
	assert(jacobian == [[1,1], [1,-2], [-3,0], [-2,-3.5], [1,1]])
