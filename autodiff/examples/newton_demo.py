'''
Demonstration of Newton's Method using our AutoDiff library
'''

def NewtonsMethod(func, x0=0, tol=1e-10,maxiters=100):
    '''
    Takes in a function, an initial guess (x0), an error tolerance,
    and a maximum number of itterations.

    Computes roots of 'func' through itterative guesses until change is below below tolerance (tol)

    func must be composed of 'AutoDiff.Operators' structures.

    '''
    xn = x0

    for i in range(maxiters):
        #Calculate y at this step.
        y = func.eval(xn)[0]
        #Calculate derivative at this step
        dy_dx = func.eval(xn)[1]

        #If y reaches tolerance, stop
        if abs(y) < tol:
            break

        #Compute Newton Step
        x_next = y / dy_dx
        #Update X
        xn = xn + x_next

        return(xn)
  
