# This file serves to test the operator.py module
from ..operators.operator import Operator as op
from ..nodes.scalar import Scalar
import numpy as np

x = Scalar(0, 1)

def test_sin():
    assert np.sin(0) == op.sin(x).val and np.cos(0) == op.sin(x).der
