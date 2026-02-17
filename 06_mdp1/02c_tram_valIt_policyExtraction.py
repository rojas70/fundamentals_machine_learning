## TODO: fix. after i introduced startingState, was clumsy and not working correctly. then removed... but

import os

### Model (MDP problem)

class TransportationMDP(object):

    def __init__(self, startingState, N):
        self.N = N
        self.startingState = startingState

    def startState(self):
        return self.startingState

    def isEnd(self, state):
        return state == self.N

    def actions(self, state):
        '''
        return list of valid actions
        '''
        result = []

        if state+1 <= self.N: result.append('walk')
        if state*2 <= self.N: result.append('tram')

        return result

    def succProbReward(self, state, action):
        '''
        This method computes and returns: (i) next states, (ii) their transition
        probabilities, and (iii) their rewards as list of tuples. 
        
        Transition probabilities p(s'|s,a)
            p(s|'walk', s) = 1.0
            
            p(s*2|'tram', s) = 0.5
            p(s  |'tram', s) = 0.5
            
        Rewards (based only on actions): R(a)            
            R('walk') = -1          (walk is more expensive)
            R('tram') = -0.25
        '''       
        result = []

        if action == 'walk':
            result.append((state+1, 1, -2)) #(s',p,r)

        elif action == 'tram':
            failProb = 0.5
            
            result.append((state*2, 1.-failProb, -1))
            result.append((state,   failProb,    -1))

        return result

    def discount(self):
        '''
        Return value of discounted reward
        '''
        return 1.

    def states(self):
        '''
        Return equivalent of vector of states. In this case, we return an iterator (range)
        '''
        return range(self.startingState, self.N+1) # Recall: range goes from startingState to (N+1)-1

# Inference (Algorithms)
def valueIteration(mdp):
    ''' 
    Use dictionaries to encode values of states. 

    1. Base case: V_0 = 0. 
    2. Use bellman equation (in this case via Q(s,a) to compute new values)

    V(s) = max_a ( SUM_s' T(s,a,s')*(R + gamma*V(s')) )
    '''

    def Q(state, action):
        ''' 
        Q-value computeation: Q(s,a) = SUM_s' T(s,a,s')*(R + gamma*V(s')) 
        '''    
        return sum(prob*(reward + mdp.discount()*V[nextState]) \
                for nextState, prob, reward in mdp.succProbReward(state, action))

    V = {}
    it = 0

    # Initialization
    for state in mdp.states():
        V[state] = 0.

    # Compute the new values (newV) given the old values (V) recursively until: |V(s')-V(s)| < epsilon for all s.
    while True:
        newV = {}
        for state in mdp.states():

            # Set done state values
            if mdp.isEnd(state): 
                newV[state] = 0.

            # Iterate: V_{k+1} = max_a () Sum T(s,a,s')[R+gV_k] )                 
            else:
                newV[state] = max(Q(state, action) for action in mdp.actions(state))
        
        # check if the convergence threshold is greater than the largest difference of any state to end
        if max( abs( V[state]-newV[state] ) for state in mdp.states() ) < 1e-10:
            break
        V = newV

        #------------------------------------------------------------------------------------
        # Policy Extraction
        # Create a dictionary pi that will provide and action for state s: \pi(s)
        # Use 2 for loops:
        # 1. Go through all states
        # 2. Go through all actions for a given state
        # Call q-function q(s,a) to retrieve the q-value. 
        # Note: 
        # - create a tuple ( q(s,a), 'action' ) with a max to keep the action of the max q-value. 
        # - Then, store the action as the value of pi[state] by selecting the 2nd element of the tuple.
        #------------------------------------------------------------------------------------
        pi = {}
        for state in mdp.states():
            
            if mdp.isEnd(state):
                pi[state] = 'none'
                
            else:
                # First compute tuple of tuples of qVals <<but append the action since we want the policy>> ((qVal1,'action1'), (qVal2,'action2'), ...)
                # Then choose the max qVal out all qVals based on the first index ([0] by default)
                # Then return only the action argmax (not the value) to return the policy (so choose the [1] index)
                pi[state] = max( ( Q(state, action), action ) for action in mdp.actions(state) )[1]

        # Print Value Iterations
        print('\n\n\n')
        #os.system('clear')
        
        print(f'Iteration {it}')
        print(f"{'s'}\t {'V(s)'}\t\t {'pi(s)'}")
        for state in mdp.states():
            print(f'{state}\t{V[state]:0.02f}\t\t {pi[state]}')

        it += 1
    return (newV, pi)

# main----------------------------------------------------------------------------------------------
# instantiate your environment (startingState, #states)
startingState = 1
mdp = TransportationMDP(startingState,N=10) # Can change start state to facilitate analysis. Start at 9/10... then 8/10. Then analyze 10 and 20 states. 

# Call Value Iteration with policy extraction
(V,pi) = valueIteration(mdp)