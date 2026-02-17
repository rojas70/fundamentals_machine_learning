def computeEditDistance(s, t):
    
    def recurse(m, n):
        """
        Return the minimum edit distance between:
        - first m letters of s
        - first n letters of t
        """
        
        # Base Case 01
        # If s is an empty string, return the length of t.
        if m == 0:  			
            cost = n
            
        # Base Case 02
        # If t is an empty string, return the length of t.
        elif n == 0:
            cost = m
            
        # When two letters are equal: "cat" vs "cat" (both length 3, but to check look at index 2)
        # - call recurse with a length decreased by 1 for both strings
        # - cost is 0
        elif s[m - 1] == t[n - 1]:  	# Last letter matches "ab" "bb"
            cost = 0 + recurse(m - 1, n - 1)
            
        # Else 3 choices: m>n, n>m, or m==n but different letters.
        # This creates a tree with different cost paths. Graph.
        else:
            # Different letters, sub in s(m), cost of 1.
            subCost = 1 + recurse(m - 1, n - 1) 	

            # m>n by 1, so delete s(m), cost of 1.            
            delCost = 1 + recurse(m - 1, n)		    # del s(m)

            # n>m by 1, so insert s(m) or equivalent del t(m), cost of 1.
            insCost = 1 + recurse(m, n - 1)		    # insert s(m): equal to a del on t(n). 

            # Take the minimum of the three costs.
            cost = min(subCost, delCost, insCost)
            
        # Finally return cost computed for any of the options above
        return cost

    # Compute the cost given the strings and their lengths
    cost = recurse(len(s), len(t)) # use lengths to start process at the end.
    return cost

## Test Cases

# # Empty string
cost = computeEditDistance('', 'a')
print(cost)

# # One letter difference
cost = computeEditDistance('a', 'a')
print(cost)

# # One letter difference
cost = computeEditDistance('b', 'a')
print(cost)

# # Length of two, same string
cost = computeEditDistance('aa', 'aa')
print(cost)

# # Length of two, one letter difference
cost = computeEditDistance('ba','aa')
print(cost)

# # Length of three, one letter difference
cost = computeEditDistance('cat','cot')
print(cost)
# Longer String
cost = computeEditDistance('a cat!', 'the cats!')
print(cost)

# Much longer
print(computeEditDistance('a cat!' * 10, 'the cats!' * 10))
a = "hello world"