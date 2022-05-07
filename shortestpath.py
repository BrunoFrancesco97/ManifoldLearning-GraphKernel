
import numpy as np
from numpy.linalg import norm


class ShortestPathKernel():
    def initialize_paths(self, G):
        INF_ = float('inf')
        v = G.shape[0]
        dist = G
        dist[dist == 0] = INF_
        np.fill_diagonal(dist, 0)
        return dist
    
    def FloydWarshall(self, G):
        G = G.astype(np.float)
        dist = self.initialize_paths(G)
        v = G.shape[0]
        for k in range(v):
            for i in range(v):
                for j in range(v):
                    dist[i,j] = min(dist[i,j], dist[i,k] + dist[k,j])
        return dist
    
    def shortest_paths(self, matrices):
        SP = []
        for el in matrices:
            SP.append(self.FloydWarshall(el))
        return SP


    def extract_freq_vector(self, S, delta):
        F = np.empty([delta+1, 1])
        for i in range(delta+1):
            F[i] = np.sum(S == i)
        return F/norm(F)
      
    def kernel_similarity(self, SP1, SP2):
        delta = int(np.maximum(np.max(SP1), np.max(SP2)))
        F1 = self.extract_freq_vector(SP1, delta)
        F2 = self.extract_freq_vector(SP2, delta)
        return  np.dot(np.transpose(F1), F2)[0]
    
    def get_similarities(self, SP_graphs):
        n = len(SP_graphs)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i,j] = self.kernel_similarity(SP_graphs[i], SP_graphs[j])
        return K
