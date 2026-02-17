#02c. gradient descent for linear regression with higher dimensions and noisy ground truth

# An example with numpy to represent vecs and matrices. You won't need to use this in your HWs.
import numpy as np

#-------------------------------------------------------------
# Generate Artificial Data
#-------------------------------------------------------------
# Normally, you want to get the data and then fit w
# We can work backwards: choose a true w, choose x's, and then create the y's from it.

# Initialization
# Create a 5D Weight vec
true_w = np.array([1,1,1,1,1])
dim = len(true_w)
epochs = 200

points=[]

# Create 10k randoþm points: rand(x), y = w.dot(x)+noise
num_of_pts=10000
for i in range(num_of_pts):
    x = np.random.randn(dim)                  # Sample dim x-points from the normal gaussian distribution

    # Create y's near ground truth but corrupted with 0-mean Gaussian noise
    y = true_w.dot(x) + np.random.randn()   

    # Create your points as a list of tuples  
    points.append((x,y))                  # 1000 (5,1) pairs

#Simplified set of points (x is 2d, y is 1d)
#points=[ (np.array([1,1]),1), (np.array([2,2]),2), (np.array([3,3]),3)]
#points=[ (np.array([1,1]),-1), (np.array([2,2]),-2), (np.array([3,3]),-3)]; d=2
    
# import pprint
# pprint.pprint(points)
    
def F(w):
    return sum((w.dot(x) - y)**2 for x, y in points) / len(points)

def dF(w):
    return sum(2*(w.dot(x) - y) * x for x, y in points) / len(points)

#-------------------------------------------------------------
# Algorithm: how we compute it (the kind of gradient descent)
#-------------------------------------------------------------
# Gradient descent

def gradientDescent(F, dF, d, n):
    w = np.zeros(d)
    eta = 0.01
    
    # Run for n epochs
    for t in range(n):
        value = F(w)
        gradient = dF(w)
        w = w - eta * gradient

        # Only print data every 10 steps
        if t%10==0:                            # If residual is zero...
            np.set_printoptions(precision=4)
            # print('iteration {}: w = {}, F(w) = {:.4f}'.format(t, w, value))
            print(f'epoch {t}: w = {w}, F(w) = {value}, gradientF = {gradient}')


# Run gradient descent on approximate linear data. 
# What do you expect the result to be?
# What is the big problem in this setup?
import timeit

gradientDescent(F,dF,dim,epochs) # Loss, gradLoss, weight dims, epochs