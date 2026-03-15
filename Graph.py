import numpy as np
import matplotlib.pyplot as plt
from collections import deque

# IMplementation of class representing undirected graph
class Graph:
    
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        self.A = np.zeros((n_nodes, n_nodes), dtype=int)  # adjacency matrix
        # Keep edges list also. We need to be able to quckly pick random edge. Only adj matrix  makes it very slow. And multiple runs take too much time
        # The memory cost is negligible compared to the time it saves
        self.edges = []  

    def add_edge(self, i, j):
        """
        Adding an edge. Ignores self-loops and repeating edges
        """
        if i == j:
            return

        if self.A[i, j] == 1:
            # Check for (j,i) is not needed as the graph is undirected so (i,j) = (j,i)
            return

        self.A[i, j] = 1
        self.A[j, i] = 1

        self.edges.append((i, j))


    def neighbors(self, i):
        """
        Returns the neighbours of a given vertix
        """
        connected = self.A[i] == 1
        return np.where(connected)[0]

   
    def degree(self, i):
        """
        Returns the degree of a given vertix
        """
        connections = self.A[i]
        return int(connections.sum())


    def average_degree(self):
        """
        Returns the average of the degrees of all vertices in the graph
        """
        degrees = self.A.sum(axis=1)
        return degrees.mean()
    
    def is_connected(self):
        #BFS
        visited = set()
        queue = deque([0])

        while queue:
            node = queue.popleft()
            if node not in visited:
                visited.add(node)
                queue.extend(self.neighbors(node))

        return len(visited) == self.n_nodes


    def remove_edge(self, i, j):
        """
        Removes an edge
        """    
        if self.A[i, j] == 0:
            return

        self.A[i, j] = 0
        self.A[j, i] = 0

        # The edge will be stored either as (i,j) or (j,i). This follows from the way we store them when adding.
        if (i, j) in self.edges:
            self.edges.remove((i, j))
        elif (j, i) in self.edges:
            self.edges.remove((j, i))

    def get_random_edge(self, rng):
        """
        Returs random edge
        """
        return self.edges[rng.integers(len(self.edges))]


def get_fully_connected(n):
    """
    Returns fully connected graph with n vertices
    """
    #Every agent connected to every other agent
    G = Graph(n)
    for i in range(n):
        for j in range(i+1, n):
            G.add_edge(i, j)
    return G

# Watts-Strogatz network
def get_small_world(n, k=4, p=0.1, seed=None):
    """
    Returns graph that resembles a Watts–Strogatz small-world network.
    Follows these steps:
    - Connect each vertix to k nearest neigbours
    - THen Rewire each edge with probability p to a random node. 
    """
    if seed is not None:
        np.random.seed(seed)

    G = Graph(n)

    # Create a ring
    # For each node, connect it to k/2 neighbours on left and on right
    for i in range(n):
        for j in range(1, k // 2 + 1):
            G.add_edge(i, (i + j) % n)
            G.add_edge(i, (i - j) % n)

    # Rewire edges
    for i in range(n):
        for j in range(1, k // 2 + 1):
            
            # rewire with probability = p
            if np.random.random() < p:

                # original neighbor in the ring
                neighbour = (i + j) % n

                # Remove the original local edge
                # We have to do it to so 
                G.remove_edge(i, neighbour)

                # Node thats not already connected to i
                candidates = [x for x in range(n) if x != i and G.A[i][x] == 0]

                # Add random candidate
                if candidates:
                    new_neighbour = np.random.choice(candidates)
                    G.add_edge(i, new_neighbour)
    return G


#Barabasi-Albert
def get_scale_free(n, m=2, seed=None):
    """
    Returns a graph that represents a Barabasi-Albert scale-free network.
    Follows these these steps:
    - Create small fully connected core
    - Then add new nodes. Each connects to m existing nodes, but the probability to connect to a given node is bigger the bigger the degree of that node is.
    """
    
    if seed is not None:
        np.random.seed(seed)
    G = Graph(n)

    # Create intital fully connected subgraph
    for i in range(m):
        for j in range(i + 1, m + 1):
            G.add_edge(i, j)

    # adding new nodes
    for new_node in range(m + 1, n):
        # Compute degrees of all existing nodes
        degrees = np.array([G.degree(i) for i in range(new_node)])
        # Convert degrees into probabilities
        probs = degrees / degrees.sum()

        # Select m target nodes with preference to the more "popular"
        targets = np.random.choice(new_node,size=m, replace=False, p=probs)
        for target in targets:
            G.add_edge(new_node, target)
    return G
