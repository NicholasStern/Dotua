from ..nodes.scalar import Scalar

'''
Initialize local variables for testing. Since these tests need to be
independent of AutoDiff, we will simulate the initalization process by calling
init_jacobian once the entire 'universe' of variables has been defined
'''
# Define scalar objects
vars = x_1, x_2 = Scalar(5), Scalar(8)
for var in vars:
    var.init_jacobian(vars)

# Define functions of the scalar objects
f_1 = x_1 + x_2
f_2 = x_2 + x_1
f_3 = x_1 - x_2
f_4 = x_2 - x_1
f_5 = x_1 * x_2
f_6 = x_2 * x_1


def test_jacobian():
    # Test jacobians of scalar primitives
    assert(x_1.get_partial(x_1) == 1)
    assert(x_1.get_partial(x_2) == 0)
    assert(x_2.get_partial(x_1) == 0)
    assert(x_2.get_partial(x_2) == 1)

    # Test jacobians of additive functions
    assert(f_1.get_partial(x_1) == 1)
    assert(f_1.get_partial(x_2) == 1)
    assert(f_2.get_partial(x_1) == 1)
    assert(f_2.get_partial(x_2) == 1)

    # Test jacobians of multiplicative functions
    assert(f_5.get_partial(x_1) == 8)
    assert(f_5.get_partial(x_2) == 5)
    assert(f_6.get_partial(x_1) == 8)
    assert(f_6.get_partial(x_2) == 5)


def test_add():
    assert(f_1.eval() == (13, 2))
    assert(f_2.eval() == (13, 2))

    # Directly check commutativity
    assert(f_1.eval() == f_2.eval())


def test_subtract():
    assert(f_3.eval() == (-3, 0))
    assert(f_4.eval() == (3, 0))


def test_multiply():
    assert(f_5.eval() == (40, 13))
    assert(f_6.eval() == (40, 13))

    # Directly check commutativity
    assert(f_5.eval() == f_6.eval())



# f2 = x - 3 - y - y
#
# def test_sub():
#     assert f2._val == -6 & f2.getDerivative(x) == 1 & f2.getDerivative == -2
#
# f3 = 1 + y - x*y - x - 3
#
# def test_mul():
#     assert f3._val == -3 & f3.getDerivative(x) == -3 & f3.getDerivative == 0
#
#
# f4 = 3/x + (x*y)/2 - 4*y
#
# def test_divide():
#     assert f4._val == -4 & f4.getDerivative(x) == -2 & f3.getDerivative == -3.5
