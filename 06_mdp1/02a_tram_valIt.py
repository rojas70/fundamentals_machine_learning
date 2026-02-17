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
            p(s+1|'walk', s) = 1.0
            
            p(s*2|'tram', s) = 0.5
            p(s  |'tram', s) = 0.5
            
        Rewards (based only on actions): R(a)            
            R('walk') = -2          (walk is more expensive)
            R('tram') = -1
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

    1. Base case: V_0 = 0: all states start with a value of zero
    2. Use bellman equation (in this case via Q(s,a) to compute new values)
        V_(k+1)(s) = max_a Q(s,a) = max_a ( \sum_s' T(s,a,s')*(R + gamma*V_k(s')) )
    
    Need to save 2 V-values: one at time k, the other at time k+1. 
    We stop if their difference V_(k+1) - V_k < threshold.
    '''

    def Q(state, action):
        ''' 
        Q-value computation: Q(s,a) = SUM_s' T(s,a,s')*(R + gamma*V(s')) 
        '''
        return sum( prob*(reward + mdp.discount()*V[nextState] ) \
                for nextState, prob, reward in mdp.succProbReward(state, action))

    V = {}
    it = 0
    
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
            
            # Iterate: V_{k+1} = max_a [ Q(s,a_1), Q(s,a_2), ..., Q(s, a_n) ]
            else:
                newV[state] = max( Q(state, action) for action in mdp.actions(state) )

        # Compute state value difference for all states. If the max abs value difference is less than the convergence threshold, you are done.
        if max( abs( V[state]-newV[state] ) for state in mdp.states() ) < 1e-10:
            break

        # Save values to memory
        V = newV

        # Print Value Iterations
        print('\n\n\n')
        #os.system('clear')
        
        print(f'Iteration {it}')
        print('s \t V(s)')

        for state in mdp.states():
            print(f'{state} \t {V[state]:0.4f}')

        it += 1
    return newV
# main----------------------------------------------------------------------------------------------
# instantiate your environment (startingState, #states)
startingState = 1
mdp = TransportationMDP(startingState, N=10) # Can change start state to facilitate analysis. Start at 9/10... then 8/10. Then analyze 10 and 20 states. 

# Test action
print(mdp.actions(8))

# Test step
print(mdp.succProbReward(3, 'tram'))

# Value Iteration
# Test with different sets of rewards: positive/negative and different criteria

# 1 +ve (Large is better)
# Fitness: walk +2, tram +1 --> Further states get highest rewards, walking always wins
# Time: tram +2, walk +1    --> Further states also win but with less prob than walk

# 2 -ve (Small is better)
# Fitness: walk -1, tram -2 --> Higher negative rewards the further you move, but state 5 has a reduce cost compared to 6 showing you can tram
# Time: tram -1, walk -2    --> This setup leads to results we would expect and correlate with search cost.
V = valueIteration(mdp)         #   Ie state 5 & 9 are best, then 4 & 8, then 2|3|7.