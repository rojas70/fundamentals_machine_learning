import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#-------------------------------------------------------------
# Step 1: Generate a Synthetic Dataset
#-------------------------------------------------------------

np.random.seed(42)  # For reproducibility

n_samples = 300
n_features = 1000  # More features than samples to induce overfitting

# Generate random features
X = np.random.randn(n_samples, n_features)

# True weights with only a few non-zero entries (sparse weights)
w_true = np.zeros(n_features)
w_true[:5] = np.random.randn(5)

# Generate target values with noise (change standard dev to increase noise)
noise = np.random.normal(0, 1, size=n_samples)
y = X @ w_true + noise

#-------------------------------------------------------------
# Step 2: Split the Dataset into Training and Test Sets
#-------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

n_train_samples, n_features = X_train.shape

#-------------------------------------------------------------
# Step 3: Implement Linear Regression Models
#-------------------------------------------------------------

# Helper functions for gradient descent
def compute_loss(X, y, w):
    """
    Computes the Mean Squared Error.
    """
    residuals = X @ w - y
    return np.mean(residuals ** 2)

def compute_gradient(X, y, w):
    """
    Computes the gradient of the MSE loss.
    """
    residuals = X @ w - y
    return 2 * X.T @ residuals / X.shape[0]

def compute_loss_reg(X, y, w, l):
    """
    Computes the Regularized Mean Squared Error (Ridge Regression).
    """
    residuals = X @ w - y
    return np.mean(residuals ** 2) + (l / 2) * np.sum(w ** 2)

def compute_gradient_reg(X, y, w, l):
    """
    Computes the gradient of the Regularized MSE loss.
    """
    residuals = X @ w - y
    return 2 * X.T @ residuals / X.shape[0] + l * w

# Gradient descent implementation with corrected validation loss
def gradient_descent(X, y, X_val, y_val, l=None, eta=0.01, num_iters=2000):
    """
    Performs gradient descent to minimize the loss function.
    
    Parameters:
    - X: Training feature matrix.
    - y: Training target vector.
    - X_val: Validation feature matrix.
    - y_val: Validation target vector.
    - l: Regularization parameter (lambda). None for non-regularized.
    - eta: Learning rate.
    - num_iters: Number of iterations.
    
    Returns:
    - w: Final weight vector.
    - train_losses: List of training loss values (without regularization term).
    - val_losses: List of validation loss values.
    """
    w = np.zeros(X.shape[1])
    train_losses = []
    val_losses = []
    
    for t in range(num_iters):
        
        if l is not None:
            # Compute the regularized loss and gradient for optimization
            loss = compute_loss_reg(X, y, w, l)
            grad = compute_gradient_reg(X, y, w, l)
            
            # For recording purposes, compute the training loss without regularization
            loss_without_reg = compute_loss(X, y, w)
        else:
            loss = compute_loss(X, y, w)
            grad = compute_gradient(X, y, w)
        
            # For recording purposes, compute the training loss without regularization
            loss_without_reg = loss  # Same as loss since there's no regularization
        
        # Update weights
        w -= eta * grad
        
        # Record the training loss without regularization
        train_losses.append(loss_without_reg)
        # Compute the validation loss without regularization
        val_loss = compute_loss(X_val, y_val, w)
        val_losses.append(val_loss)
        
        # Optional: Print progress every 100 iterations
        if (t + 1) % 100 == 0:
            print(f'Iteration {t + 1}/{num_iters}, Training Loss: {loss_without_reg:.4f}, Validation Loss: {val_loss:.4f}')
    
    return w, train_losses, val_losses

#-------------------------------------------------------------
# Step 4: Train Models and Compare Performance
#-------------------------------------------------------------

# Training without regularization
print("Training without regularization:")
w_noreg, train_losses_noreg, val_losses_noreg = gradient_descent(
    X_train, y_train, X_test, y_test, l=None, eta=0.01, num_iters=2000
)

# Training with regularization
lambdas = [0.1, 1, 10, 100,1000]
best_lambda = None
best_w_reg = None
best_val_loss = float('inf')
val_losses_reg_models = {}

for l in lambdas:
    print(f"\nTraining with regularization (lambda = {l}):")
    w_reg, train_losses_reg, val_losses_reg = gradient_descent(
        X_train, y_train, X_test, y_test, l=l, eta=0.01, num_iters=2000
    )
    val_losses_reg_models[l] = val_losses_reg
    final_val_loss = val_losses_reg[-1]
    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        best_lambda = l
        best_w_reg = w_reg
        best_train_losses_reg = train_losses_reg
        best_val_losses_reg = val_losses_reg


#-------------------------------------------------------------
# Step 5: Evaluate and Compare the Models
#-------------------------------------------------------------

# Final evaluation on the test set
mse_noreg = compute_loss(X_test, y_test, w_noreg)
mse_reg = compute_loss(X_test, y_test, best_w_reg)

print("\nFinal MSE on test set:")
print(f"Without regularization: {mse_noreg:.4f}")
print(f"With regularization (lambda = {best_lambda}): {mse_reg:.4f}")

# Plotting training and validation loss curves
plt.figure(figsize=(12, 6))

# Plot for the model without regularization
plt.subplot(1, 2, 1)
plt.plot(train_losses_noreg, label='Training Loss')
plt.plot(val_losses_noreg, label='Validation Loss')
plt.title('Without Regularization')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()

# Plot for the best regularized model
plt.subplot(1, 2, 2)
plt.plot(best_train_losses_reg, label='Training Loss')
plt.plot(best_val_losses_reg, label='Validation Loss')
plt.title(f'With Regularization (lambda = {best_lambda})')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()
