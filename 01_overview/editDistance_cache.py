def computeEditDistance(s, t):
    '''
    In this version we add a cache, such that we do not have to recompute something we have seen before.
    '''

    # Use a cache to save compute. If two previously seen strings appear, simply return the pre-computed result.
    # A dictionary will be good here, as we can use a tuple of two strings as the key, and the result as the value.
    cache = {}  # (m, n) => result
    
    def recurse(m, n):
        """
        Return the minimum edit distance between:
        - first m letters of s
        - first n letters of t
        - note that m and n serve as indeces for the strings
        - strings s and t still availble as this function is within the scope of their defintion
        - can use recurssion in substitute/delete/insert calls
        """
        if (m, n) in cache:
            return cache[(m, n)]
            
        if m == 0:  			# Base case "" vs "aa"
            result = n
            
        elif n == 0:  			# Base case "aa" vs ""
            result = m
            
        elif s[m - 1] == t[n - 1]:  	# Last letter matches "ab" "bb"
            result = recurse(m - 1, n - 1)
            
        # When both entries are not the same: 3 choices
        # Other scenarios: swap/delete/insert
        # Need to add cost of 1
        # Need to decrease one or both indeces        
        else:
            subCost = 1 + recurse(m - 1, n - 1) 	# substitution
            delCost = 1 + recurse(m - 1, n)		# deletion
            insCost = 1 + recurse(m, n - 1)		# insertion
            result = min(subCost, delCost, insCost)
            
        # Cache cost and return
        cache[(m, n)] = result
        return result

    return recurse(len(s), len(t)) # compute the cost given the strings and their lengths

## Test Cases

# Empty string
computeEditDistance('', 'a')

# One letter difference
computeEditDistance('b', 'a')

# Length of two, same string
computeEditDistance('a', 'a')

# Length of two, one letter difference
computeEditDistance('cat','at')

# Longer String
computeEditDistance('a cat!', 'the cats!')

# Much longer: cache helps here considerably!
print(computeEditDistance('a cat!' * 10, 'the cats!' * 10))