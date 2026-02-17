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
    Performs stochastic gradient descent optimization with adaptive eta.

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
        The learning rate. If set to 0, an adaptive learning rate will be used.

    Returns
    -------
    w : ndarray
        The optimized weight vector after performing stochastic gradient descent.

    Notes
    -----
    - If `eta` is set to 0, an adaptive learning rate is used, which decreases over time according to `eta = 1 / sqrt(numUpdates)`, where `numUpdates` is the number of updates performed.
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
    
    # Set adaptive learning rate flag
    adaptiveEta = (eta == 0)    

    # Number of epochs
    for epoch in range(num_of_epochs):
        
        # Shuffle the dataset indices
        indices = np.random.permutation(numPoints)

        # Walk through the entire number of points in sections of size mini-batch.
        # Your points will be sectioned off as: 1st batch [0:batchSZ], 2nd batch: [batchSZ+1:2*batchSZ]... i.e. [0:32],[32:64]
        for start_idx in range(0, numPoints, batchSZ):
            
            # Determine the end index of the mini-batch
            end_idx = min(start_idx + batchSZ, numPoints)

            # Get the indices for the current mini-batch
            batch_indices = indices[start_idx:end_idx]

            # Initialize sum of gradients and function values
            gradient_sum = np.zeros_like(w)
            value_sum = 0.0

            # Compute the sum of gradients and function values over the mini-batch
            for pt in batch_indices:
                value_sum    += f(w, pt)
                gradient_sum += df(w, pt)

            # Compute average gradient and function value over the mini-batch
            batch_size_actual = end_idx - start_idx  # In case the last batch is smaller
            
            gradient_avg = gradient_sum / batch_size_actual
            value_avg    = value_sum    / batch_size_actual

            # Update learning rate if adaptive
            if adaptiveEta:
                numUpdates += 1
                eta = 1.0 / math.sqrt(numUpdates)

            # Update weights using the average gradient
            w = w - eta * gradient_avg

        # Optionally, compute and print the loss and dloss over the entire dataset for monitoring
        error  = np.mean( [ f(w, pt) for pt in range(numPoints)] )
        derror = np.mean( [df(w, pt) for pt in range(numPoints)] )
        
        print(f'Epoch {epoch + 1}/{num_of_epochs}: w = {w}, F(w) = {error:.3f}, dF = {derror}')

    return w

# Test how quickly gradientDescent and SGD converge to the correct W's.
# Compare weights and losses.

print("\n\n01. Gradient Descent:")
gradientDescent(trainLoss, gradientTrainLoss, initialWeightVector)

print("\n02. SGD:")
mini_batch = 1
stochasticGradientDescent(loss, gradientLoss, mini_batch, numPoints, initialWeightVector,eta=0.1)

print("\n03. SGD with adaptive eta:")
mini_batch = 1
stochasticGradientDescent(loss, gradientLoss, mini_batch, numPoints, initialWeightVector,eta=0.0)

mini_batch = 32
print(f"\n04.1 SGD with mini-bacth of size {mini_batch} and adaptive eta:")
stochasticGradientDescent(loss, gradientLoss, mini_batch, numPoints, initialWeightVector,eta=0.0)

mini_batch = 64
print(f"\n04.2 SGD with mini-bacth of size {mini_batch} and adaptive eta:")
stochasticGradientDescent(loss, gradientLoss, mini_batch, numPoints, initialWeightVector,eta=0.0)

mini_batch = 128
print(f"\n04.3 SGD with mini-bacth of size {mini_batch} and adaptive eta:")
stochasticGradientDescent(loss, gradientLoss, mini_batch, numPoints, initialWeightVector,eta=0.0)