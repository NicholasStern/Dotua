from autodiff.autodiff import AutoDiff as ad
from autodiff.operators.operator import Operator as op

scalars = ad.create_scalar([1, 1, 1])

x = scalars[0]
y = scalars[1]
z = scalars[2]

f_1 = x + 3 + y
print(f_1.eval())
print(f_1.partial(x))
print(f_1.partial(y))
print(f_1.partial(z))

f_2 = x * 3 + y - 2 * z
print(f_2.eval())
print(f_2.partial(x))
print(f_2.partial(y))
print(f_2.partial(z))

#f_3 = x / y - 3 * z - 4 / x
#print(f_3.eval())
#print(f_3.partial(x))
#print(f_3.partial(y))
#print(f_3.partial(z))

f_4 = op.sin(x + y) - op.cos(x / 3)
print(f_4.eval())
print(f_4.partial(x))
print(f_4.partial(y))
print(f_4.partial(z))

#f_5 = x ** 2 + 3 ** y - x ** y
#print(f_5.eval())
#print(f_5.partial(x))
#print(f_5.partial(y))
#print(f_5.partial(z))

#f = 1/(x*y*z) + op.sin(1/x + 1/y + 1/z)
#print(f.eval())
#print(f.partial(x))
#print(f.partial(y))
#print(f.partial(z))

# Define a list of vector variables
vectors = ad.create_vector([[1,2], [3,4]],2)

# Define name for the vector variables you get
x = vectors[0]
y = vectors[1]

# Define scalar functions about vectors
# Warning: basic operations such as '+' '-' '*' '/' '**' are not supported between elemenets from different vectors
# e.g. You can define functions such as f = x[0] - x[1], but you can not define functions such as f = x[0] + y[1]
# Basic operations are supported between vectors
# e.g. You can define functions such as f = x ** y or g = x - y + z or h = f + g as long as x,y,z are all vector variables
f_1 = x[0] + x[1]
f_2 = x[0] - 3 - x[1] - x[1]
f_3 = 1 + x[1] - x[0] * x[1] - x[0] - 3
f_4 = 3 / x[0] + (x[0] * x[1]) / 2 - 4 * x[1]
f_5 = x[0] ** 3 + x[1] / x[0]
f_6 =  2 ** x[1]

# Define vector function to the same vector
f = [f_1, f_2, f_3, f_4, f_5, f_6]
jacobian = []

# You can get jacobian matrix of this vector function to vector
for function in f:
	jacobian.append(function.eval()[1])

print(jacobian)

