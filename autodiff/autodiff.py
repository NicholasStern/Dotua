from autodiff.nodes.scalar import Scalar
from autodiff.nodes.vector import Vector


class AutoDiff():
    @staticmethod
    def create_scalar(vals):
        '''
        Returns a list of Scalar variables to the user,
        with the values initialized to the user defined values or all 0
        by default
        '''

        try:
            scalars = [None] * len(vals)
            for i in range(len(vals)):
                scalars[i] = Scalar(vals[i])

            # Initialize the jacobians
            for var in scalars:
                var.init_jacobian(scalars)

            return vars
        except TypeError:
            scalar = Scalar(vals)
            scalar.init_jacobian([scalar])
            return scalar

    @staticmethod
    def create_vector(vals, num=1):
        '''

        INPUTS
        ======
        vals: list of lists of floats, compulsory
            Value of the list of Vector variables
        num: int, compulsory, default value is 1
            Number of vector variables in the list
        RETURNS
        ========
        A list of Vector variables

        NOTES
        =====
        PRE:
            - the length of vals sould be num
        POST:
            returns a list of vector variables with value defined in vals
        '''
        if(num > len(vals)):
            print('You want to create a list of {} vectors'.format(num))
            print('But you only put values for the first {} vectors'.format(len(vals)))
        elif(num < len(vals)):
            print('You want to create a list of {} vectors'.format(num))
            print('But you only put values for {} vectors'.format(len(vals)))
        vars = [None] * num
        for i in range(num):
            try:
                vars[i] = Vector(vals[i])
            except Exception:
                vars[i] = Vector([0])
        return vars
