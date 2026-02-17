import numpy as np
#02a.Regularized Gradient Descent for linear regression intro

# Data
#-------------------------------------------------------------
# A good way to create simple points is a list of tuples. 
#-------------------------------------------------------------
points = [(1, 2), (2, 1), (3, 2), (4, 3), (5,2), (6,3), (7,2), (8,3), (9,2)] #(x, y) Best line will have a negative gradient of [del_x,del_y.]
#-------------------------------------------------------------
d = len(points)

def F(w,l=None):
	'''
	Non-regularized training loss (Mean Squared Error).
	'''
	return sum((w * x - y)**2 for x, y in points) / len(points)

def dF(w,l=None):
	'''
    Gradient of the non-regularized training loss with respect to w.	'''
	return sum(2*(w * x - y) * x for x, y in points) / len(points)

def FR(w,l):
    '''
    Regularized training loss (MSE + L2 regularization).
    '''
    mse = sum( (w*x - y)**2 for x,y in points) / len(points)
    regularization = (l/2)*w**2
    return mse + regularization

def dFR(w,l):
    '''
	Gradient of the regularized training loss with respect to w.
	'''
    gradient_mse = sum(2*(w * x - y) * x for x, y in points) / len(points)
    gradient_regularization = l*w
    return gradient_mse + gradient_regularization

#-------------------------------------------------------------
# Algorithm: how we compute it (the kind of gradient descent)
#-------------------------------------------------------------
# Gradient descent
def gradientDescent(F, dF,l=None):
    '''
    Performs gradient descent to minimize the loss function F.
    '''
    
    # Initial weight and step size
    w = 0
    eta = 0.01
    print(f'Value of lambda: {l}')
    
    # Epochs
    for t in range(100):
        
        # Loss and gradient loss
        loss = F(w,l)
        dLoss = dF(w,l)

        # Optimization
        w = w - eta*dLoss
        
        # Print every 20 epochs
        if t % 20 == 0:
            print( f'iteration {t}: w = {w:.3f}, F(w) = {loss:.4f}' )
    
    print('\n')
    return w, loss

#---- Callilng functions
# Do gradient descent.
# It helps to set training loss and its derivative as functions (more flexible code)
print('Regular Gradient Descent: ')
gradientDescent(F,dF)

print('\nRegularized Gradient Descent: ')
min = 100
best_coeff = -1

cLambda = [0,0.5,1,5,10,100]
for l in cLambda:
    _,v=gradientDescent(FR,dFR,l)
    if np.abs(v) < np.abs(min):
        min = v
        best_coeff = l
print(f'A value of lambda = {best_coeff}, yields the lowest loss.')

# ----
# Plot
import matplotlib.pyplot as plt

lambda_values = [0, 0.1, 0.5, 1, 5, 10]
losses = []

for l in lambda_values:
    _, loss = gradientDescent(FR, dFR, l=l)
    losses.append(loss)

plt.plot(lambda_values, losses, marker='o')
plt.xlabel('Lambda')
plt.ylabel('Training Loss')
plt.title('Training Loss vs. Regularization Parameter')
plt.show()
