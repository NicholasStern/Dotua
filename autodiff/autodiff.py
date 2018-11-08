import autodiff.nodes.scalar as scalar


class AutoDiff():
    def __init__(self):
        pass

    @staticmethod
    def create_scalar(vals):
        '''
        Returns a list of Scalar variables to the user,
        with the values initialized to the user defined values or all 0
        by default
        '''

        vars = [None] * 
        for i in range(num):
            try:
                vars[i] = scalar.Scalar(vals[i])
            except TypeError:
                vars[i] = scalar.Scalar(vals)

        # Initials the jacobians for the scalars
        for var in vars:
            var.init_jacobian(vars)
        return vars
