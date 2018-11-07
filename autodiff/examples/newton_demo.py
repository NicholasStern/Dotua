def NewtonsMethod(func, x0, tol=1e-5,maxiters=100):
    '''
    Computes the roots of func through iterative guesses until change is below tolerance.
    
    Takes in a function (func), an initial guess (x0), an error tolerance (tol), and a maximum number of iterations (maxiters).
    
    func must be composed of 'AutoDiff.Operators' structures.
    
    Example Usage
    -------------
    
        $ import autodiff as ad
        $ x = ad.create_scalar(0)
        $ from autodiff.operators import Operator as op
        $ func = op.sin(x)
        $ NewtonsMethod(y,0)
        
        >>> 1

    '''
    xn = x0

    for i in range(maxiters):
        
        #Calculate y at this step.
        y = f(xn)
        ##y = func.eval(xn)[0]
        #Calculate derivative at this step
        dy_dx = df(xn)
        ##dy_dx = func.eval(xn)[1]

        #If y reaches tolerance, stop
        if abs(y) < tol:
            return(xn)
            break
        
        else:
            #Compute Newton Step
            x_next = y / dy_dx
            #Update X
            xn = xn + x_next
            
            return(xn)