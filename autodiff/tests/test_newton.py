from ..operators.operator import Operator as op
from ..nodes.scalar import Scalar
from .. import autodiff as ad
from ..examples import newton_demo as newton


#define example function using ad objects
x = ad.create_scalar(0)
func = op.sin(x)

def test_newton_method():
    assert newton.NewtonsMethod(func,0) == np.cos(0)
    