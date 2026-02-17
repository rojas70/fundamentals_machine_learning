'''
The task is to predict whether a particular input string contains a name 
or not. 

Files: 
    - main_simple.py: script to develop and run
    - util.py: utilities to help you run your code
    - names.train and names.dev: files to train/validate

We have train/validate/test sets: name.train, name.dev, name.test
    - Each set consists of one string and one binary label +1 or -1. 

There will be a 'util.py' script, that will help you do a number of things:
    1. automatically read and generate your training and validation data sets
        trainExamples = util.readExamples('names.train')
    2. Ouput weight and error information. I.e.:
        util.outputWeights(weights, 'weights')
        util.outputErrorAnalysis(devExamples, featureExtractor, weights, 'error-analysis')

Goal:
    1. Learn a predictor based on:
        Use your train and validation sets
        A feature extractor that you have to design

    2. Based on your results and analysis, iterate to make better

    3. Finally, test on test data set and report your loss/error there

Feature Extractor:
    Remember, from ML.7 we looked at feature templates.
    - Feature templates are sparse by nature.
    - As such they are best represented by dictionaries.
    - Instead of using a standard dictionary, try a default dict.
    - When a default dict is used, and a key that does not exist is called, 
      the defultdict will yield a value according to the type passed. 

      Ie a = defaultdict(float)
      then >>  a['b']
      the dict automatically creates: a={b:0.0}... works for all standard python types

Predictor:
    You will want to learn a classifier
    Utilize the same hinge loss you have already learned
    The difference is now you must apply it to feature templates (i.e. to dictionaries)      
        - Make sure you are familiar with some of the main methods of the dict class. 
        a = dict() or defaultdict(float)
        a['a']=1, a['b']=2
        a.keys() or a.values() or a.items()
        and running loops or list comprehensions from these
        for k,v in a.items().... 

        - The above is used to create a weight dictionary with the same keys as for the input features and to do their dot product for example


#---------------------------
The Development Cycle
#---------------------------
1. Making sense of the data
    - Notice that every string may have stuff to the left and to the right of the name.
      We can think of this information as context. 
    - Start with a null feature extractor
    - Run main_simple.py to run a simple sanity check

2. Create first simple feature extractor
    a) feature template for entity
    b) check train/validation error. 
    c) then do weight/error analysis.
        test error high?
        what are some predictions wrong?
        what could you improve?

3. Improve Feature extractor
    a) Try to improve feature extractor
    b) Repeat iteratively
'''