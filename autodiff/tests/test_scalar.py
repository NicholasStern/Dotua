from ..nodes.scalar import Scalar

x = Scalar(1)
y = Scalar(2)

f1 = x + 3 + y

def test_add():
    assert f1._val == 6 & f1.getDerivative(x) == 1 & f1.getDerivative(y) == 1

f2 = x - 3 - y - y

def test_sub():
    assert f2._val == -6 & f2.getDerivative(x) == 1 & f2.getDerivative == -2

f3 = 1 + y - x*y - x - 3

def test_mul():
    assert f3._val == -3 & f3.getDerivative(x) == -3 & f3.getDerivative == 0


f4 = 3/x + (x*y)/2 - 4*y

def test_divide():
    assert f4._val == -4 & f4.getDerivative(x) == -2 & f3.getDerivative == -3.5
