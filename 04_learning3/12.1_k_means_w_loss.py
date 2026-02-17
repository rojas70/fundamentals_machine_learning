import random
import math

def euclidean_distance(point1, point2):
    '''
    Inputs:
    - point1: regular point to test
    - point2: centroid
    
    Output:
    - euclidean_distance
    
    Extract x's and y's by zipping: zip:(x1,y1),(x2,y2)-->(x1,x2),(y1,y2)
    Then compute euclidean distance as sqrt( (x2-x1)^2 + (y2-y1)^2 )
    '''
    return math.sqrt( sum( (p1 - p2) ** 2 for p1, p2 in zip(point1, point2) ) ) 

# Function to calculate the mean of a list of points to yield a centroid
def calculate_mean(points):
    '''
    Inputs:
    - points: list of lists: [ [x1,y1], [x2,y2], ... ]
    
    Outputs: 
    - list of n-dim mean values [ x_mean, y_mean ]
    
    Computes the mean values across dimensions.
    '''
    
    num_points = len(points)
    num_dimensions = len( points[0] ) # i.e. 2 dimensional (x1,x2)?
    
    # Create n-dim list to hold mean values. 
    mean = [0] * num_dimensions
    
    # Compute avg: sum / #pts
    for point in points:
        
        # For each dimension, accumulate values
        for i in range(num_dimensions):
            mean[i] += point[i]
            
    # Returns list of means across dimensions
    return [ mean[i] / num_points for i in range(num_dimensions) ]

# Function to check if centroids have converged (within tolerance)
def has_converged(old_centroids, new_centroids, tolerance=1e-5):
    
    # Pair up sets of (x,y) coords in old and new centroids
    for old, new in zip(old_centroids, new_centroids):
        
        # Pair up x-values and y-values 
        for old_val, new_val in zip(old, new):
            
            # If absolute difference across dims is less than a tolerance error for ALL pairs, you have converged
            if abs(old_val - new_val) > tolerance:
                return False
    return True

def computeClusterLoss(cluster_points,centroid):
    # Inputs: 
    # - cluster_points: points that pertain to specific clusters
    # - centroid: list of centroids
    # Output:
    # - loss: returns the sum of the losses for each cluster
    #
    # Compute the squared difference between poitns and centroid for a cluster
    ptLoss = []
    
    for pt in cluster_points:        
        # Accumulate L2_norm distances within clusteR: \sum norm(pt - \mu)^2 = \sum ( (y-mu_y)^2 + (x-mu_x)^2 )
        ptLoss.append( sum( ((pt[dim] - centroid[dim])**2) for dim in range(len(centroid)) ) )

    # Loss for a given cluster
    return sum(ptLoss)     

# K-means algorithm
def k_means(X, K, max_iters=100):
    '''
    Inputs: 
    - X: list of points
    - K: number of clusters
    - max_iters: maximum number of iterations
    
    Outputs:
    - centroids: list of K centroids
    - assignments: list of assignments for each point
    - loss: sum of the losses across all clusters    
    '''
    
    # Step 0: Initialize K centroids randomly from the data points
    centroids = random.sample(X, K) # choose K random elements from X. i.e. random.sample([1,2,3,4],2)

    for t in range(max_iters):
        
        #--- Step 1: Expectation. Assign each point to the closest centroid ---#
        
        # Init:
        # list of assignments for each point. length = total num  points.
        assignments = []  
        
        # Go through each point
        for point in X:
            
            # Compute the eucliden distance and place in list distances: [x_dist, y_dist]
            distances = [euclidean_distance(point, centroid) for centroid in centroids]
            
            # Given one point and all centroids, compute min distance. min distance corresponds to some index. Assign it to point 
            assignments.append( distances.index( min(distances) ) )  # The index() method for lists in Python is used to find the index (position) of the first occurrence of a specified element in the list.

        #--- Step 2: Maximization. Update centroids by calculating the mean of the points in each cluster ---#
        
        # Init
        # list of new centroids
        new_centroids = []
        
        # Collect all points with the same assignment -> cluster
        cluster_points = []
        
        # Loss: within cluster_sum_of_squares
        clusterLoss = []
        
        # For each cluster: 0,1,2,...
        for k in range(K):
            
            # Go through all the points assigned to a given cluster
            #cluster_points = [ X[i] for i in range(len(X)) if assignments[i] == k ] # in one line
            for i in range(len(X)): 
                if assignments[i] == k:
                    cluster_points.append(X[i]) # accumulate the point
            
            
            # Avoid empty clusters
            if cluster_points:  
                
                # Compute the new mean of those points and set to new centroids
                new_centroids.append(calculate_mean(cluster_points))
                
            else:
                new_centroids.append(centroids[k])  # Keep the same centroid if the cluster is empty
                
            clusterLoss.append( computeClusterLoss(cluster_points,new_centroids[k]) )
            cluster_points = []  # Reset for next cluster

        # Compute the total loss at every iteration to keep track of progress
        loss = sum(clusterLoss)
        
        # Print the loss at each iteration
        print(f"Iteration {t+1}, loss (within cluster sum of squares): {loss}")

        # Check for convergence (if centroids do not change)
        if has_converged(centroids, new_centroids):            
            break
        
        centroids = new_centroids

    return centroids, assignments, loss

# Example usage:
if __name__ == "__main__":
    
    # Step 0: Create a simple dataset with two clusters in 2D
    # Use lists vs tuples for mutability and simplicity in handling updates (tuples are immutable)
    X = [
        [1.0, 1.0], [1.5, 1.5], [0.8, 1.0], [1.2, 1.2],  # Cluster 1
        [5.0, 5.0], [5.5, 5.2], [4.8, 4.9], [5.2, 5.1]   # Cluster 2
    ]

    # Hyperparameters
    K = 2       # Number of clusters
    iters = 5   # Number of iterations

    # Run the k-means algorithm. Returns centroids and assignments.
    final_centroids, final_assignments, loss = k_means(X, K, iters)

    # Output the results
    print("Final centroids: ")
    for centroid in final_centroids:
        print(centroid)

    print("\nPoint assignments: ")
    for i, assignment in enumerate(final_assignments):
        print(f"Point {X[i]} is assigned to cluster {assignment + 1}")
