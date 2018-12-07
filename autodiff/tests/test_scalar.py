from ..nodes.scalar import Scalar
import numpy as np

'''
Initialize local variables for testing. Since these tests need to be
independent of AutoDiff, we will simulate the initalization process by calling
init_jacobian once the entire 'universe' of variables has been defined
'''
# Define scalar objects
vars = x, y = Scalar(1), Scalar(2)
a, b = x.eval()[0], y.eval()[0]
for var in vars:
    var.init_jacobian(vars)

# Define functions of the scalar objects
f_1 = x + y
f_2 = y + x
f_3 = x - y
f_4 = y - x
f_5 = x * y
f_6 = y * x
f_7 = x / y
f_8 = y / x

# Slightly more complicated functions
g_1 = 10 * x + y / 2 + 1000
g_2 = -2 * x * x - 1 / y

# Exponential functions
h_1 = x ** 2
h_2 = 2 ** x
h_3 = x ** y


def test_jacobian():
    # Test jacobians of scalar primitives
    assert x.eval()[1] == {x: 1, y: 0}
    assert y.eval()[1] == {x: 0, y: 1}

    # Test jacobians of functions with addition
    assert f_1.eval()[1] == {x: 1, y: 1}
    assert f_2.eval()[1] == {x: 1, y: 1}

    # Test jacobians of functions with subtraction
    assert f_3.eval()[1] == {x: 1, y: -1}
    assert f_4.eval()[1] == {x: -1, y: 1}

    # Test jacobians of functions with multiplication
    assert f_5.eval()[1] == {x: 2, y: 1}
    assert f_6.eval()[1] == {x: 2, y: 1}

    # Test jacobians of functions with division
    assert f_7.eval()[1] == {x: 1/2, y: -1/4}
    assert f_8.eval()[1] == {x: -2, y: 1}

    # Test jacobians of more complicated functions
    assert g_1.eval()[1] == {x: 10, y: 1/2}
    assert g_2.eval()[1] == {x: -4, y: 1/4}

    # Test jacobians for exponentials and deg > 1 polynomials
    assert h_1.eval()[1] == {x: 2 * a, y: 0}
    assert h_2.eval()[1] == {x: (2 ** a) * np.log(2), y: 0}


def test_add():
    assert f_1.eval()[0] == a + b
    assert f_2.eval()[0] == a + b

    # Directly check commutativity
    assert f_1.eval() == f_2.eval()

    # Check addition with a constant
    radd = 5 + x
    assert radd.eval()[0] == 5 + a


def test_subtract():
    assert f_3.eval()[0] == a - b
    assert f_4.eval()[0] == b - a

    # Check subtraction from a constant
    radd = 5 - x
    assert radd.eval()[0] == 5 - a


def test_multiply():
    assert f_5.eval()[0] == a * b
    assert f_6.eval()[0] == a * b

    # Directly check commutativity
    assert f_5.eval() == f_6.eval()


def test_divide():
    assert f_7.eval()[0] == a / b
    assert f_8.eval()[0] == b / a


def test_power():
    assert h_1.eval()[0] == a ** 2
    assert h_2.eval()[0] == 2 ** a
    assert h_3.eval()[0] == a ** b


def test_other():
    assert g_1.eval()[0] == 10 * a + b / 2 + 1000
    assert g_2.eval()[0] == -2 * (a ** 2) - 1 / b
