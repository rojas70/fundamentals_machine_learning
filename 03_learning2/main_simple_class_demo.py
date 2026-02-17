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
    
    phi = defaultdict(float)
    tokens = x.split()  # creates a list of substrings. i.e. " 3. Tommi Makinen (" --> [3., Tommi, Makinen, (]"

    # Keep the name
    left, entity, right = tokens[0], tokens[1:-1], tokens[-1]
    
    # feature template
    phi['entity is: ' + ' '.join(entity)] = 1
    
    # Individual Names
    for name in entity:
        phi['entity is: ' + name] = 1  
        
        # Prefix
        phi['entity is: ' + name[:3]] = 1  
        
        # Suffix
        phi['entity is: ' + name[-3:]] = 1  

    # Leverage context
    phi['entity is: ' + left] = 1
    phi['entity is: ' + right] = 1
    
    # Contrast vs generic case
    # for word in tokens:
    #     phi['entity is: ' + word] += 1        
    
    return phi  

# Learn a predictor
eta = 0.01
epochs = 10 
weights = submission.learnPredictor(trainExamples, devExamples, featureExtractor, epochs, eta)
util.outputWeights(weights, 'weights')
util.outputErrorAnalysis(devExamples, featureExtractor, weights, 'error-analysis')

# Test!!!
testExamples = util.readExamples('names.test')
predictor = lambda x : 1 if util.dotProduct(featureExtractor(x), weights) > 0 else -1
print(f"----------------------------------------\nTest Error = \
     {util.evaluatePredictor(testExamples, predictor)} \
     \n----------------------------------------")