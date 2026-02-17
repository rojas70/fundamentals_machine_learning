import numpy as np

# Data
#-------------------------------------------------------------
# Now with two features
#-------------------------------------------------------------
# Feature matrix X of shape (n_samples, n_features)
X = np.array([
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

# True weights (for generating synthetic target values)
w_true = np.array([0.5, -0.2])

# Target vector y with some added noise
np.random.seed(0)  # For reproducibility
y = X @ w_true + np.random.normal(0, 0.5, size=X.shape[0])

n_samples, n_features = X.shape

#-------------------------------------------------------------
# Loss Functions and Their Gradients
#-------------------------------------------------------------
def F(w):
    """
    Non-regularized training loss (Mean Squared Error).
    """
    residuals = X @ w - y
    return np.mean(residuals ** 2)

def dF(w):
    """
    Gradient of the non-regularized training loss with respect to w.
    """
    residuals = X @ w - y
    return 2 * X.T @ residuals / n_samples

def FR(w, l):
    """
    Regularized training loss (MSE + L2 regularization).
    """
    residuals = X @ w - y
    return np.mean(residuals ** 2) + (l / 2) * np.sum(w ** 2)

def dFR(w, l):
    """
    Gradient of the regularized training loss with respect to w.
    """
    residuals = X @ w - y
    return 2 * X.T @ residuals / n_samples + l * w

#-------------------------------------------------------------
# Gradient Descent Algorithm
#-------------------------------------------------------------
def gradient_descent(F, dF, w0=None, eta=0.01, l=None, num_iters=100):
    """
    Performs gradient descent to minimize the loss function F.
    
    Parameters:
    - F: The loss function.
    - dF: The derivative of the loss function.
    - w0: Initial weight vector.
    - eta: Learning rate.
    - l: Regularization parameter (lambda). None for non-regularized.
    - num_iters: Number of iterations.
    
    Returns:
    - The final weight vector w.
    - The final loss value after gradient descent.
    """
    if w0 is None:
        w = np.zeros(n_features)
    else:
        w = w0.copy()
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
            print(f'Iteration {t}: w = {w}, Loss = {loss:.4f}')
    print('\n')
    return w, loss

#-------------------------------------------------------------
# Running Gradient Descent
#-------------------------------------------------------------
# Non-regularized gradient descent
print('Gradient Descent without Regularization:')
w_final, loss_final = gradient_descent(F, dF)

# Regularized gradient descent with different lambda values
print('Gradient Descent with Regularization:')
min_loss = float('inf')
best_lambda = None
best_w = None
lambda_values = [0, 0.1, 0.5, 1, 5, 10]

for l in lambda_values:
    w_final, loss = gradient_descent(FR, dFR, l=l)
    if loss < min_loss:
        min_loss = loss
        best_lambda = l
        best_w = w_final

print(f'A value of lambda = {best_lambda} yields the lowest loss of {min_loss:.4f}.')
print(f'The best weight vector is {best_w}')
