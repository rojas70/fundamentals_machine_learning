import numpy as np

#02b. gradient descent for linear regression with a bias/intercept
#points = [ (1,1), (2,2), (3,3) ] # Exact fit
points = [( 1,1), (2,3), (4,3)] # class notes
#points = [( 1,2), (2,2), (3,6)] # pull it down


def phi(x):
    '''
    Features: in this case add a bias term to the input to model y-intercept
    '''
    p=np.array( [1, x] )
    return p

def trainLoss(w):
    '''
    train loss... presented in an easier to digest form, but less efficient... 
    - changed the names up a bit
    - not using comprehension/python way. Done to simplify the visualization of the debugging process here.
    '''
    s=0
    for x, y in points:
        score = w.dot(phi(x))
        residual = (score - y)**2
        s += residual
    return 1/len(points) * s

# def trainLoss(w):
    # return sum((w.dot( phi(x) )-y)**2 for x,y in points) / len(points)

def gradTrainLoss(w):
        return 1/len(points) * sum( 2 * (w.dot(phi(x)) - y) * phi(x) for x, y in points )

# define gradientDescent. points available globally.
def gradientDescent(F, dF, dim):
    ''' 
    1. init weights
    2. init eta
    for some num of iterations
    3. compute trainLoss
    4. compute gradTrainLoss
    5. do gradient descent
    6. report performance
    '''

    w = np.zeros(dim)
    eta = 0.1

    for t in range(200):
        tloss = F(w)
        grad_tloss = dF(w)
        w = w - eta * grad_tloss
        #print( 'epoch: {:0.2f}, TrainLoss: {:0.3f}, Weights: {:0.3f}'.format(t,tloss,w) )

        # Print with adjusted precision
        np.set_printoptions(precision=3) # adjusts w
        print( f'epoch: {t}, TrainLoss: {tloss:0.3f}, Weights: {w}' )


# main
dim = 2                                         # Two weights for gradient and offset
gradientDescent(trainLoss, gradTrainLoss, dim)  # Could pass epoch and points in here too