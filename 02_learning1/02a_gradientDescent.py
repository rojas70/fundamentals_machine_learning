#02a. gradient descent for linear regression intro

# Data
#-------------------------------------------------------------
# A good way to create simple points is a list of tuples. 
#-------------------------------------------------------------
#points = [(1, 1), (2, 3), (4, 3)] #(x, y) Best line will have a negative gradient of [delta_x,delta_y.]
points = [(1, 1), (2, 2), (3, 3)]
#points = [(1, 1), (2, 4), (4, 16)]
#-------------------------------------------------------------

def trainLoss(w):
	'''
	Training loss: 
	'''
	return sum( (w * x - y)**2 for x, y in points ) / len(points)

def dtrainLoss(w):
	'''
	minimize training loss by computing the gradient: 
	'''
	return sum( 2*(w * x - y) * x for x, y in points ) / len(points)

#-------------------------------------------------------------
# Algorithm: how we compute it (the kind of gradient descent)
#-------------------------------------------------------------
# Gradient descent

def gradientDescent(F, dF):

	# Initialize weights: set to zero for simplicity
	w = 0

	# Learning rate Hyperparameter: set by user. small value desirable. Try playing with the value during training.
	eta = 0.01

	# Epochs: Number of Gradient Steps (Hyperparameter)
	for epoch in range(50):

		# Update error info
		value = F(w)			# At the end of the epoch, we want to know the value of the loss, hoping it converges to zero
		gradient = dF(w)		# Use gradient value to update weights

		# Update weight info
		w = w - eta * gradient

		# Print data
		print(f'Epoch {epoch}: w = {w:0.4f}, training loss = {value:0.4f}')

# Do gradient descent.
# It helps to set training loss and its derivative as functions (more flexible code)
gradientDescent(trainLoss,dtrainLoss)