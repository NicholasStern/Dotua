import autodiff.nodes.scalar as scalar

class AutoDiff():
    def __init__(self):
        pass

    @staticmethod
    def create_scalar(num=1, vals=[0]):
        '''
        Returns a list of Scalar variables to the user,
        with the values initialized to the user defined values or all 0
        by default
        '''

        vars = [None]*num
        for i in range(num):
            try:
                vars[i] = scalar.Scalar(vals[i])
            except IndexError:
                vars[i] = scalar.Scalar()
        return vars

    @staticmethod
    def create_vector(num, vals):
        '''
        The idea is similar to create_scalar.
        This will allow the user to create vectors and specify initial
        values for the elements of the vectors.
        '''
        pass
