from nodes import scalar
from nodes import vector


class AutoDiff():
    def __init__(self):
        pass

    @staticmethod
    def create_scalar(self, num=1, vals=[0]):
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
    def create_vector(self, num, vals):
        '''
        The idea is similar to create_scalar.
        This will allow the user to create vectors and specify initial
        values for the elements of the vectors.
        '''
        pass
