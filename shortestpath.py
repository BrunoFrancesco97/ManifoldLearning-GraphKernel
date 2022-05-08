
import numpy as np
from numpy import linalg

class ShortestPathKernel():
    def init(self, G):
        INF_ = float('inf')
        v = G.shape[0]
        dist = G
        dist[dist == 0] = INF_
        np.fill_diagonal(dist, 0)
        return dist
    
    def FloydWarshall(self, graph):
        dist = self.init(graph)
        v = graph.shape[0]
        for k in range(v):
            for i in range(v):
                for j in range(v):
                    dist[i,j] = min(dist[i,j], dist[i,k] + dist[k,j])
        return dist
    
    def shortest_paths(self, matrices):
        shortest_paths = []
        for el in matrices:
            shortest_paths.append(self.FloydWarshall(el))
        return shortest_paths


    def extract_freq_vector(self, S, delta):
        freq = np.empty([delta+1, 1])
        for i in range(delta+1):
            freq[i] = np.sum(S == i)
        return freq/linalg.norm(freq)
      
    def kernel_similarity(self, path1, path2):
        delta = int(np.maximum(np.max(path1), np.max(path2)))
        F1 = self.extract_freq_vector(path1, delta)
        F2 = self.extract_freq_vector(path2, delta)
        return  np.dot(np.transpose(F1), F2)[0]
    
    def get_similarities(self, shortest_paths):
        n = len(shortest_paths)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i,j] = self.kernel_similarity(shortest_paths[i], shortest_paths[j])
        return K
