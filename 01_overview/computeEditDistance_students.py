# Defintion
def computeEditDistance(s,t):

    # Create caching dict
    d={}

    if (s,t) in d:
        return d[(s,t)]
    
    def recurse(m,n):
        # base cases

        # Emptry strings
        if m==0:
            return n
        
        elif n==0:
            return m
        
        # Both entries are the same
        elif s[m-1]==t[n-1]:
            cost = recurse(m-1,n-1)
        
        # compute the cost of swapping, deleting, and inserting and compute the min cost
        else:
            # swaping
            swap_cost = 1 + recurse(m-1,n-1)

            # delete entry at s
            del_cost = 1 + recurse(m-1,n)

            # insert at s
            ins_cost = 1 + recurse(m,n-1)

            cost = min(swap_cost, del_cost, ins_cost)

        return cost

    cost = recurse(len(s),len(t))
    d[ (s,t) ] = cost
    return cost

# -- Calls
s = ""
t = ""
cost = computeEditDistance(s,t)
print(f"The cost of recurse is {cost}")