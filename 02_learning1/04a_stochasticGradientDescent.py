# 04. gradient descent for linear regression with stochastic gradient descent
import math
import numpy as np
np.set_printoptions(precision=4)

# Data -----------------------------------------------------------
numPoints = 10000

# Epochs ------------------------------------------------------------
num_of_epochs = 10

# Set the ground truth
trueW = np.array([1, 2, 3, 4, 5])

# Create a generative function
def generate():
    x = np.random.randn(len(trueW))
    y = trueW.dot(x) + np.random.randn()
    #print('example', x, y)
    return (x, y)

# Generate Training Examples
trainExamples = [generate() for i in range(numPoints)] 

# Model -----------------------------------------------------------
def phi(x):
    return np.array(x)

def initialWeightVector():
    '''
    Return a numpy array of len trueW filled with zeros of type float
    '''
    return np.zeros(len(trueW), dtype=float)

 # Regular Gradient Descent

def trainLoss(w):
    return 1.0 / len(trainExamples) * sum((w.dot(phi(x)) - y)**2 for x, y in trainExamples)

def gradientTrainLoss(w):
    return 1.0 / len(trainExamples) * sum(2 * (w.dot(phi(x)) - y) * phi(x) for x, y in trainExamples)

# SGD Methods ------------------------------------------------------------

def loss(w, i):
    x, y = trainExamples[i]
    return (w.dot(phi(x)) - y)**2

def gradientLoss(w, i):
    x, y = trainExamples[i]
    return 2 * (w.dot(phi(x)) - y) * phi(x)

# Optimization -----------------------------------------------------------
def gradientDescent(F, dF, initialWeightVector):
    w = initialWeightVector()
    eta = 0.1

    # Update weights using all training examples (gradient of train loss)
    for epoch in range(num_of_epochs):
        loss = F(w)
        loss.astype(float)     # hack to facilitate printing of value
        
        gradient = dF(w)
        w = w - eta * gradient
        
        # Print data every epoch
        np.set_printoptions(precision=3) # adjusts w
        print(f'epoch {epoch}: w = {w}, F(w) = {loss:.3f}, dF = {gradient}')

def stochasticGradientDescent(f, df, batchSZ, numPoints, weightFunc,eta=0.1):
    '''
    Performs stochastic gradient descent optimization. 
    In this implementation, each epoch will:
     - update the gradient using a single randomized point 
     - Update the gradient for all the points.
    In effect, if you have numPoints, you will have numPoints gradient updates per epoch vs 1.

    Parameters
    ----------
    f : callable
        The loss function to minimize. It should accept the weight vector `w` and a data point index `pt` as inputs and return a scalar loss value.
    dF : callable
        The gradient of the loss function. It should accept the weight vector `w` and a data point index `pt` as inputs and return the gradient vector with respect to `w`.
    batchSZ : int
        The size of the mini-batch used in each iteration.
    numPoints : int
        The total number of data points in the dataset.
    initialWeightVector : callable
        A function that returns the initial weight vector `w`.
    eta : float, optional (default=0.1)
        The learning rate. 

    Returns
    -------
    w : ndarray
        The optimized weight vector after performing stochastic gradient descent.

    Notes
    -----
    - The function runs for a fixed number of epochs (10 by default), updating the weights based on the average gradient computed over randomly selected mini-batches.
    - This implementation supports both standard stochastic gradient descent (when `batchSZ=1`) and mini-batch gradient descent (when `batchSZ > 1`).

    Examples
    --------
    >>> def f(w, pt):
    ...     # Example loss function
    ...     return np.sum(w**2)
    >>> def dF(w, pt):
    ...     # Gradient of the example loss function
    ...     return 2 * w
    >>> initialWeightVector = lambda: np.zeros(5)
    >>> w_opt = stochasticGradientDescent(f, dF, batchSZ=10, numPoints=100, initialWeightVector=initialWeightVector)
    '''
    
    # Initalize local variables: weight vector and number of updates
    w = weightFunc()
    numUpdates = 0

    # Number of epochs
    # Epochs: rounds of calculations
    for epoch in range(num_of_epochs):
    # Updates to w according to the batch size
    # note: points will update for each epoch
    #
    # - standard SGD: use all data points (randomized) -- one at a time per epoch
    # - miniBatch: use a subset of points in each epoch -- one at a time
        points = np.random.randint(0, numPoints, numPoints)
        
        # Update weights according to randomly selected training points
        for pt in points:
            error = f(w, pt)
            derror = df(w, pt)

            # Udpate weight
            w = w - eta * derror

        print(f'epoch {epoch}/{num_of_epochs}: w = {w}, F(w) = {error:.3f}, dF = {derror}')

    return w

# Test how quickly gradientDescent and SGD converge to the correct W's.
# Compare weights and losses.

print("\n\n01. Gradient Descent:")
gradientDescent(trainLoss, gradientTrainLoss, initialWeightVector)

print("\n02. SGD:")
miniBatch = 1
stochasticGradientDescent(loss, gradientLoss, miniBatch, numPoints, initialWeightVector,eta=0.1)