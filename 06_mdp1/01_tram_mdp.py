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

        # Conditions for acting: walk/tram
        if state+1 <= self.N: result.append('walk')
        if state*2 <= self.N: result.append('tram')

        return result

    def succProbReward(self, state, action):
        '''
        This method computes transitions and rewards. 
        
        Assuming the transition conditional probability is known,
        given a state + all possible actions --> new states will be computed 

        Additionally, we assume we know the rewards, which are also given when R(s,a,s') a transition occurs.

        The method will return a list of result tuples which contain (newState, prob, reward) 
        '''       
        result = []

        if action == 'walk':
            result.append((state+1, 1, -2))

        elif action == 'tram':
            failProb = 0.5
            result.append( (state*2, 1.-failProb, -1) )
            result.append( (state,    failProb,   -1) )

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
        return range(self.startingState, self.N+1)

# main -----------------

# instantiate your environment
startingState=8
mdp = TransportationMDP(startingState, N=10)

# Test action
print(mdp.actions(8))

# Test step
print(mdp.succProbReward(3,'tram'))
