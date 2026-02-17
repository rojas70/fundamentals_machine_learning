import numpy as np

# Data
#-------------------------------------------------------------
# Using NumPy arrays for efficient computations
#-------------------------------------------------------------
points = np.array([
    [1, 2],
    [2, 1],
    [3, 2],
    [4, 3],
    [5, 2],
    [6, 3],
    [7, 2],
    [8, 3],
    [9, 2]
])

X = points[:, 0]
y = points[:, 1]
n = len(y)

#-------------------------------------------------------------
# Loss Functions and Their Gradients
#-------------------------------------------------------------
def F(w):
    """
    Non-regularized training loss (Mean Squared Error).
    """
    return np.mean((w * X - y) ** 2)

def dF(w):
    """
    Gradient of the non-regularized training loss with respect to w.
    """
    return 2 * np.mean((w * X - y) * X)

def FR(w, l):
    """
    Regularized training loss (MSE + L2 regularization).
    """
    return np.mean((w * X - y) ** 2) + (l / 2) * w ** 2

def dFR(w, l):
    """
    Gradient of the regularized training loss with respect to w.
    """
    return 2 * np.mean((w * X - y) * X) + l * w

#-------------------------------------------------------------
# Gradient Descent Algorithm
#-------------------------------------------------------------
def gradient_descent(F, dF, w0=0, eta=0.01, l=None, num_iters=100):
    """
    Performs gradient descent to minimize the loss function F.
    
    Parameters:
    - F: The loss function.
    - dF: The derivative of the loss function.
    - w0: Initial weight.
    - eta: Learning rate.
    - l: Regularization parameter (lambda). None for non-regularized.
    - num_iters: Number of iterations.
    
    Returns:
    - The final loss value after gradient descent.
    """
    w = w0
    print(f'Value of lambda: {l}')
    
    for t in range(num_iters):
        if l is not None:
            loss = F(w, l)
            grad = dF(w, l)
        else:
            loss = F(w)
            grad = dF(w)
        w -= eta * grad
        if t % 20 == 0:
            print(f'Iteration {t}: w = {w:.4f}, Loss = {loss:.4f}')
    print('\n')
    return w,loss

#-------------------------------------------------------------
# Running Gradient Descent
#-------------------------------------------------------------
# Non-regularized gradient descent
print('Gradient Descent without Regularization:')
gradient_descent(F, dF)

# Regularized gradient descent with different lambda values
print('Gradient Descent with Regularization:')
min_loss = float('inf')
best_lambda = None
lambda_values = [0, 0.5, 1, 5, 10, 100]

for l in lambda_values:
    _,loss = gradient_descent(FR, dFR, l=l)
    if loss < min_loss:
        min_loss = loss
        best_lambda = l

print(f'A value of lambda = {best_lambda} yields the lowest loss of {min_loss:.4f}.')

# Plot
import matplotlib.pyplot as plt

lambda_values = [0, 0.1, 0.5, 1, 5, 10]
losses = []

for l in lambda_values:
    _, loss = gradient_descent(FR, dFR, l=l)
    losses.append(loss)

plt.plot(lambda_values, losses, marker='o')
plt.xlabel('Lambda')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. Regularization Parameter')
plt.show()
