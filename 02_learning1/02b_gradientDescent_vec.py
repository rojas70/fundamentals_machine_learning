#02b. gradient descent for linear regression using numpy arrays

# An example with numpy to represent vecs and matrices. You won't need to use this in your HWs.
# To create vectors with numpy do: np.array(...). 	0D vector is: np.array([]) 		-> np.shape()=(n,)
												  # 1D vector is: np.array([[]])	-> np.shape()=(1,n)
import numpy as np								  # 2D vector is: np.array([[],[]]) -> np.shape()=(2,n)


# Labeled Data
#-------------------------------------------------------------
# Instead of using a number for x, represent x as a 1D vector and y for label: ([x], y)
#-------------------------------------------------------------
points = [ (np.array([2]),4), (np.array([4]),2) ] # 1D array for each pair
#points = [ (np.array([2]),-2), (np.array([3]),-3) ] 

def F(w):
    return sum((w.dot(x) - y)**2 for x, y in points) / len(points)

def dF(w):
    return sum(2*(w.dot(x) - y) * x for x, y in points) / len(points)

#--------------------------------------------------------------
# Optimization via # Gradient descent
#--------------------------------------------------------------

def gradientDescent(F, dF, d):

	# Initialize Parameters
	w = np.zeros(d)
	eta = 0.01
      
	# Updates weights over a number of epochs
	for epoch in range(100):
            
		# Output error info
		value = F(w)
		gradient = dF(w)
            
		# Update weight info
		w = w - eta * gradient
        
		# Set print options for numpy values for array w
		np.set_printoptions(precision=4) 
		print( f'epoch: {epoch}, TrainLoss: {value:0.2f}, Weights: {w}' )

#--- Main --------------- 
dim = 1                                         # Single weight for gradient, no offset.
gradientDescent(F,dF,dim)
