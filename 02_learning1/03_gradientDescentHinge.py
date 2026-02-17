# 03. gradient descent for linear classification with a hinge loss
import numpy as np

# Create training data which consist of 2D x-data and a ground-truth y label as a list of tuple (x, y) pairs

# Class boundary around x=y axis
trainExamples = [
    ((0,  2),  1), 
    ((-2, 0),  1),
    ((1, -1), -1),   
]

# Boundary around x=0
# trainExamples = [
#     ((-1, 0), -1), 
#     ((-2, 1), -1),
#     ((-3, 2), -1),

#     ((1, 1),  1),
#     ((2, -1), 1),
#     ((3, -2), 1),    
# ]
# Remember w will be the vector normal to the dividing line/plane
#------------------------------------------------------------------------------------------------------------------
def phi(x):
    return np.array(x)

def initialWeightVector():
    return np.zeros(2)

def trainLoss(w):
    '''The objective is defined as the max between the downslope and the 0-function'''

    return sum( max( 1 - w.dot(phi(x)) * y, 0 ) for x, y in trainExamples ) / len(trainExamples)

def gradientTrainLoss(w):
    return sum( -phi(x) * y if 1 - w.dot(phi(x) ) * y > 0 else 0 for x, y in trainExamples) / len(trainExamples)

#---------------------------------------------------------
# Optimization algorithm
#---------------------------------------------------------
def gradientDescent(F, gradientF, initialWeightVector):
    w = initialWeightVector()
    eta = 0.1
    for t in range(100):
        value = F(w)
        gradient = gradientF(w)
        w = w - eta * gradient
        np.set_printoptions(precision=4)
        print(f'epoch {t}: F(w) = {value}, gradientF = {gradient}, w = {w}')

#------------------------------------------------------------------------------------------------------------------
# call gradient descent function for classification, using the hingeloss
gradientDescent(trainLoss, gradientTrainLoss, initialWeightVector)
