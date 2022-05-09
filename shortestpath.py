
import numpy as np
from numpy.linalg import norm

class ShortestPathKernel():
    def FloydWarshall(self, graph):
        dist = graph
        dist[dist == 0] = float('inf')
        np.fill_diagonal(dist, 0)
        dim = graph.shape[0]
        for k in range(dim):
            for i in range(dim):
                for j in range(dim):
                    dist[i,j] = min(dist[i,j], dist[i,k] + dist[k,j])
        return dist
    
    def shortestPaths(self, matrices):
        shortestPaths = []
        for el in matrices:
            shortestPaths.append(self.FloydWarshall(el))
        return shortestPaths


    def getFrequencyVector(self, path, delta):
        freq = np.empty([delta+1, 1])
        for i in range(delta+1):
            freq[i] = np.sum(path == i)
        res = freq/norm(freq)
        return res
      
    def deltaKernel(self, path1, path2):
        delta = int(np.maximum(np.max(path1), np.max(path2)))
        freq1 = self.getFrequencyVector(path1, delta)
        freq2 = self.getFrequencyVector(path2, delta)
        return  np.dot(np.transpose(freq1), freq2)[0]
    
    def getSimilarities(self, shortestPaths):
        n = len(shortestPaths)
        similarities = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                similarities[i,j] = self.deltaKernel(shortestPaths[i], shortestPaths[j])
        return similarities
