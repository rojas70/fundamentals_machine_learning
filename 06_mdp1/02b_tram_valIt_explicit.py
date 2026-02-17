import os
import sys
sys.setrecursionlimit(1000)
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
    Use dictionaries to encode values of states: V[s] = v

    1. Base case: V_0 = 0. 
    2. Use bellman equation (in this case via Q(s,a) to compute new values)
        V(s) = max_a ( \sum_s' T(s,a,s')*(R + gamma*V(s')) )
    
    Need to save 2 V-values: one at time k, the other at time k+1. 
    We stop if their difference V_(k+1) - V_k < threshold.
    '''

    def Q(state, action):
        ''' 
        Q-value computation: Q(s,a) = SUM_s' T(s,a,s')*(R + gamma*V(s')) 
        '''
        q = []
        for nextState, prob, reward in mdp.succProbReward(state, action):
            q.append( prob*( reward + mdp.discount()*V[nextState] ))
        return sum(q) 

    V = {}
    it = 0
    qNode = []

    # Base-case: k=0, bottom of tree, initialize all state values to zero
    for state in mdp.states():
        V[state] = 0.                

    # Compute the new values (newV) given the old values (V) recursively until: |V(s')-V(s)| < epsilon for all s.
    while True:
        
        newV = {}

        # Do Bellman update for all states
        for state in mdp.states():

            # Set reward for reaching final state
            if mdp.isEnd(state): 
                newV[state] = 0.
            
            # Iterate: V_{k+1} = max_a () Sum T(s,a,s')[R+gV_k] ) 
            else:
                for action in mdp.actions(state):
                    qNode.append( Q(state, action)  )
                newV[state] = max( qNode )

                # Reset qNode values
                qNode.clear()

        # check for convergence
        if max( abs( V[state]-newV[state] ) for state in mdp.states() ) < 1e-10:
            break

        V = newV

        # Print Value Iterations
        print('\n\n\n')
        #os.system('clear')
        
        print(f'Iteration {it}')
        print('{:5} {:5}'.format('s', 'V(s)'))

        for state in mdp.states():
            print('{:5} {:5}'.format(state, V[state]))

        it += 1   
    return newV     

# main ----------------------------------------------------------------------------------------------
# instantiate your environment
mdp = TransportationMDP(8,N=10)

# Test action
# print(mdp.actions(8))

# Test step
#print(mdp.succProbReward(3, 'tram'))

# Value Iteration
V = valueIteration(mdp)
