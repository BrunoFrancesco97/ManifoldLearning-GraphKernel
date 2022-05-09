
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
    
    def shortestPaths(self, adjMatrices):
        shortestPaths = []
        for el in adjMatrices:
            shortestPaths.append(self.FloydWarshall(el))
        return shortestPaths


    def getFrequencyVector(self, paths, delta):
        freq = np.empty([delta+1, 1])
        for i in range(delta+1):
            freq[i] = np.sum(paths == i)
        res = freq/norm(freq)
        return res
      
    def deltaKernel(self, paths1, paths2):
        delta = int(np.maximum(np.max(paths1), np.max(paths2)))
        freq1 = self.getFrequencyVector(paths1, delta)
        freq2 = self.getFrequencyVector(paths1, delta)
        return  np.dot(np.transpose(freq1), freq2)[0]
    
    def getSimilarities(self, shortestPathGraphs):
        n = len(shortestPathGraphs)
        similarities = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                similarities[i,j] = self.deltaKernel(shortestPathGraphs[i], shortestPathGraphs[j])
        return similarities
