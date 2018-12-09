from rautodiff.rnodes.rscalar import rScalar

class rAutoDiff():
    def __init__(self):
        self.func = None

    @staticmethod
    def create_rscalar(vals):
        '''
        Return rScalar object(s) with user defined value(s).

        INPUTS
        ======
        vals: list of lists of floats, compulsory
            Value of the list of Vector variables

        RETURNS
        ========
        A list of Vector variables

        NOTES
        =====

        POST:
            returns a list of vector variables with value defined in vals
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
        '''
        Reset attributes of variables to make sure it works for the new function

        INPUTS
        =====
        var: list of variables
        '''
        try:
            for i in range(len(var)):
                var[i].parents = []
                var[i].grad_val = None
        except TypeError:
            var.parents = []
            var.grad_val = None

    def partial(self, func, var):
        '''
        Returns the gradient of the function with regarding to this variable

        INPUTS
        =====
        func: a function of rscalar variable
        var: a rscalar variable

        RETURNS
        =======
        A constant, which is the gradient of func with regarding to var
        '''
        func.grad_val = 1
        return var.gradient()
