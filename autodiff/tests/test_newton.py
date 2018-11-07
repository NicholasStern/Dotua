from ..operators.operator import Operator as op
from ..nodes.scalar import Scalar
from ..examples import newton_demo as newton
import numpy as np

#define example function using ad objects
x = Scalar(0,1)
func = op.sin(x)

def test_newton_method():
    assert newton.NewtonsMethod(func,0) == np.cos(0)
