# 04. Gradient Descent for Linear Regression with Stochastic Gradient Descent
import math
import numpy as np
np.set_printoptions(precision=4)
import matplotlib.pyplot as plt  # Import matplotlib for plotting

# Data -----------------------------------------------------------
numPoints = 10000

# Epochs ------------------------------------------------------------
num_of_epochs = 100

# Set the ground truth weights
trueW = np.array([1, 2, 3, 4, 5])

# Create a generative function
def generate():
    x = np.random.randn(len(trueW))
    y = trueW.dot(x) + np.random.randn()
    return (x, y)

# Generate Training Examples
trainExamples = [generate() for i in range(numPoints)]

# Model -----------------------------------------------------------
def phi(x):
    return np.array(x)

def initialWeightVector():
    '''
    Return a numpy array of length equal to trueW filled with zeros of type float
    '''
    return np.zeros(len(trueW), dtype=float)

# Regular Gradient Descent Functions --------------------------------
def trainLoss(w):
    return 1.0 / len(trainExamples) * sum((w.dot(phi(x)) - y)**2 for x, y in trainExamples)

def gradientTrainLoss(w):
    return 1.0 / len(trainExamples) * sum(2 * (w.dot(phi(x)) - y) * phi(x) for x, y in trainExamples)

# SGD Functions -----------------------------------------------------
def loss(w, i):
    x, y = trainExamples[i]
    return (w.dot(phi(x)) - y)**2

def gradientLoss(w, i):
    x, y = trainExamples[i]
    return 2 * (w.dot(phi(x)) - y) * phi(x)

# Abstracted Plot Function ------------------------------------------
def plot_errors_subplot(ax, errors_list, labels, markers, title):
    '''
    Plots the errors on a given subplot axis.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The subplot axis to plot on.
    errors_list : list of lists
        Each element is a list of errors for a particular run.
    labels : list of str
        Labels for each run.
    markers : list of str
        Marker styles for each run.
    title : str
        Title of the subplot.
    '''
    for errors, label, marker in zip(errors_list, labels, markers):
        ax.plot(range(1, num_of_epochs + 1), errors, marker=marker, label=label)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)

# Optimization Functions --------------------------------------------
def gradientDescent(F, dF, initialWeightVector):
    w = initialWeightVector()
    eta = 0.1
    errors = []  # List to store loss values at each epoch

    # Update weights using all training examples (gradient of train loss)
    for epoch in range(num_of_epochs):
        loss_value = F(w)
        gradient = dF(w)
        w = w - eta * gradient

        # Record error
        errors.append(loss_value)

        # Print data every epoch
        np.set_printoptions(precision=3)  # Adjusts printing precision for weights
        print(f'epoch {epoch + 1}: w = {w}, F(w) = {loss_value:.3f}, dF = {gradient}')

    return w, errors  # Return weights and errors

def stochasticGradientDescent(f, df, batchSZ, numPoints, weightFunc, eta=0.1):
    '''
    Performs stochastic gradient descent optimization with optional adaptive eta.

    Parameters
    ----------
    f : callable
        The loss function to minimize.
    df : callable
        The gradient of the loss function.
    batchSZ : int
        The size of the mini-batch used in each iteration.
    numPoints : int
        The total number of data points in the dataset.
    weightFunc : callable
        A function that returns the initial weight vector `w`.
    eta : float, optional (default=0.1)
        The learning rate. If set to 0, an adaptive learning rate will be used.

    Returns
    -------
    w : ndarray
        The optimized weight vector after performing stochastic gradient descent.
    errors : list
        List of loss values at each epoch.
    '''
    # Initialize variables
    w = weightFunc()
    numUpdates = 0
    errors = []  # List to store loss values at each epoch

    # Set adaptive learning rate flag
    adaptiveEta = (eta == 0)

    # Number of epochs
    for epoch in range(num_of_epochs):
        # Shuffle the dataset indices
        indices = np.random.permutation(numPoints)

        # Process data in mini-batches
        for start_idx in range(0, numPoints, batchSZ):
            # Determine the end index of the mini-batch
            end_idx = min(start_idx + batchSZ, numPoints)
            batch_indices = indices[start_idx:end_idx]

            # Initialize sum of gradients and function values
            gradient_sum = np.zeros_like(w)
            value_sum = 0.0

            # Compute the sum of gradients and function values over the mini-batch
            for pt in batch_indices:
                value_sum += f(w, pt)
                gradient_sum += df(w, pt)

            # Compute average gradient and function value over the mini-batch
            batch_size_actual = end_idx - start_idx  # In case the last batch is smaller
            gradient_avg = gradient_sum / batch_size_actual
            value_avg = value_sum / batch_size_actual

            # Update learning rate if adaptive
            if adaptiveEta:
                numUpdates += 1
                eta = 1.0 / math.sqrt(numUpdates)

            # Update weights using the average gradient
            w = w - eta * gradient_avg

        # Compute loss over the entire dataset for monitoring
        error = np.mean([f(w, pt) for pt in range(numPoints)])
        errors.append(error)  # Record error at the end of each epoch

        print(f'Epoch {epoch + 1}/{num_of_epochs}: w = {w}, F(w) = {error:.3f}')

    return w, errors  # Return weights and errors

# Main Script -------------------------------------------------------
# Test how quickly gradientDescent and SGD converge to the correct W's.

import matplotlib.pyplot as plt

# Initialize subplots
fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# First Plot: Gradient Descent
print("\n\n01. Gradient Descent:")
w_gd, errors_gd = gradientDescent(trainLoss, gradientTrainLoss, initialWeightVector)

# Plotting the loss for Gradient Descent
errors_list = [errors_gd]
labels = ['Gradient Descent']
markers = ['o']
plot_errors_subplot(axs[0, 0], errors_list, labels, markers, 'Gradient Descent Loss per Epoch')

# Second Plot: SGD
print("\n02. SGD:")
mini_batch = 1
w_sgd, errors_sgd = stochasticGradientDescent(loss, gradientLoss, mini_batch, numPoints, initialWeightVector, eta=0.1)

# Plotting the loss for SGD
errors_list = [errors_sgd]
labels = ['SGD (Batch Size 1, eta=0.1)']
markers = ['s']
plot_errors_subplot(axs[0, 1], errors_list, labels, markers, 'SGD Loss per Epoch')

# Third Plot: SGD with Adaptive Eta
print("\n03. SGD with adaptive eta:")
mini_batch = 1
w_sgd_adaptive, errors_sgd_adaptive = stochasticGradientDescent(loss, gradientLoss, mini_batch, numPoints, initialWeightVector, eta=0.0)

# Plotting the loss for SGD with adaptive eta
errors_list = [errors_sgd_adaptive]
labels = ['SGD (Batch Size 1, Adaptive Eta)']
markers = ['^']
plot_errors_subplot(axs[1, 0], errors_list, labels, markers, 'SGD with Adaptive Eta Loss per Epoch')

# Fourth Plot: SGD with Different Mini-Batch Sizes
mini_batch_sizes = [32, 64, 128]
errors_list = []
labels = []
markers = ['o', 's', '^']

print("\n04. SGD with different mini-batch sizes and adaptive eta:")
for mini_batch, marker in zip(mini_batch_sizes, markers):
    print(f"\nSGD with mini-batch of size {mini_batch}:")
    w_sgd_mb, errors_sgd_mb = stochasticGradientDescent(loss, gradientLoss, mini_batch, numPoints, initialWeightVector, eta=0.0)
    errors_list.append(errors_sgd_mb)
    labels.append(f'Batch Size {mini_batch}')

# Plotting the losses for different mini-batch sizes
plot_errors_subplot(axs[1, 1], errors_list, labels, markers, 'SGD with Adaptive Eta and Different Mini-Batch Sizes')

# Adjust layout and display all plots
plt.tight_layout()
plt.show()
