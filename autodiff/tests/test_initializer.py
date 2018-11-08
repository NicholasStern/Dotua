'''
This file tests that the AutoDiff initializer functions defined in autodiff.py
are correct.
'''
from .. import autodiff as ad


def test_single_variable():
    x = ad.AutoDiff.create_scalar(vals=[5])[0]
    assert(x._val == 5)
