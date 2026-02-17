import submission, util
from collections import defaultdict

# Read in examples
# data organized into 3 string segments: left context, entity, right context
trainExamples = util.readExamples('names.train')
devExamples = util.readExamples('names.dev')

# Development cylce

def featureExtractor(x):  # phi(x)
    '''
    Create a sparse dict of input strings x. 
    Values are counts for words
    '''
    pass    

# Learn a predictor
weights = submission.learnPredictor(trainExamples, devExamples, featureExtractor, 10, 0.01)
util.outputWeights(weights, 'weights')
util.outputErrorAnalysis(devExamples, featureExtractor, weights, 'error-analysis')

# Test!!!
testExamples = util.readExamples('names.test')
predictor = lambda x : 1 if util.dotProduct(featureExtractor(x), weights) > 0 else -1
print('test error =', util.evaluatePredictor(testExamples, predictor))
