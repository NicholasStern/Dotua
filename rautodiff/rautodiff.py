from rautodiff.rnodes.rscalar import rScalar

class rAutoDiff():
    def __init__(self):
        universe = []
        func = None

    @staticmethod
    def create_rscalar(vals):
        '''
        Returns a list of Scalar variables to the user,
        with the values initialized to the user defined values or all 0
        by default
        '''
        try:
        	rscalars = [None] * len(vals)
        	for i in range(len(vals)):
        		rscalars[i] = rScalar(vals[i])
        	return rscalars
        except TypeError:
        	rscalar = rScalar(vals)
        	return rscalar

    def reset_universe(self, var):
    	try:
    		for i in range(len(var)):
    			var[i].parents = []
    			var[i].grad_val = None
    	except TypeError:
    		var.parents = []
    		var.grad_val = None

    def partial(self, func, var):
    	func.grad_val = 1
    	return var.gradient()
