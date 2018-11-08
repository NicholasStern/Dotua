import autodiff.nodes.scalar as scalar
import autodiff.nodes.vector as vector


class AutoDiff():
    def __init__(self):
        pass

    @staticmethod
    def create_scalar(vals=[0], num=1):
        '''
        Returns a list of Scalar variables to the user,
        with the values initialized to the user defined values or all 0
        by default
        '''

        vars = [None]*num
        for i in range(num):
            try:
                vars[i] = scalar.Scalar(vals[i])
            except TypeError:
                vars[i] = scalar.Scalar(vals)

        # Initials the jacobians for the scalars
        for var in vars:
            var.init_jacobian(vars)
        return vars

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
                vars[i] = vector.Vector(vals[i])
            except Exception:
                vars[i] = vector.Vector([0])
        return vars
