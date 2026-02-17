import submission, util
from collections import defaultdict

# Read in examples
trainExamples = util.readExamples('names.train')
devExamples = util.readExamples('names.dev')

def featureExtractor(x):  # phi(x)
    # x = "took Mauritius into"

    #1. structure for template features?
    phi=defaultdict(float)

    # Split input text into tokens
    tokens = x.split()
    
    # Ready to begin to build features. 
    
    # Round 1: 
    # Try to extract names and call it entity:
    left,entity,right=tokens[0],tokens[1:-1],tokens[-1]     # Split token
    phi['entity is: ' + ' '.join(entity) ]=1                # Build feature vector

    # Round 2: integrate additional contextual tokens: left and right
    phi['left is: ' + ''.join(left)]      = 1   
    phi['right is: ' + ''.join(right)]    = 1


    #4 Add sub-tokens from name
    for word in entity:
        phi['word in name is: ' + word] = 1
    
    #5 suffix/prefix
        phi['prefix in name is: ' + word[:4]] = 1
        phi['suffix in name is: ' + word[-4:]] = 1
    return phi

# Learn a predictor
weights = submission.learnPredictor(trainExamples, devExamples, featureExtractor,100,0.01)
util.outputWeights(weights, 'weights')
util.outputErrorAnalysis(devExamples, featureExtractor, weights, 'error-analysis')

# Test!!!
testExamples = util.readExamples('names.test')
predictor = lambda x : 1 if util.dotProduct(featureExtractor(x), weights) > 0 else -1
print('test error =', util.evaluatePredictor(testExamples, predictor))
