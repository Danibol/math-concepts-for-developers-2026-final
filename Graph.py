import numpy as np

class Graph:
    
    def __init__(self, n_nodes):
        self.n_nodes = n_nodes
        self.A = np.zeros((n_nodes, n_nodes), dtype=int)  # adjacency matrix
    
    def add_edge(self, i, j):
        self.A[i][j] = 1
        self.A[j][i] = 1
    
    def neighbors(self, i):
        connected = self.A[i] == 1
        return np.where(connected)[0]

    def degree(self, i):
        connections = self.A[i]
        return int(connections.sum())

    def average_degree(self):
        degrees = self.A.sum(axis=1)
        return degrees.mean()
    
    def is_connected(self):
        # BFS
        visited = set()
        queue = [0]
        while queue:
            node = queue.pop(0)
            if node not in visited:
                visited.add(node)
                queue.extend(self.neighbors(node))
        return len(visited) == self.n_nodes



def get_fully_connected(n):
    #Every agent connected to every other agent
    G = Graph(n)
    for i in range(n):
        for j in range(i+1, n):
            G.add_edge(i, j)
    return G