import numpy as np
from numpy.linalg import norm


class ShortestPathKernel():
    def FloydWarshall(self, graph):
        shortpath = graph
        shortpath[shortpath == 0] = float('inf')
        np.fill_diagonal(shortpath, 0)
        dim = graph.shape[0]
        for k in range(dim):
            for i in range(dim):
                for j in range(dim):
                    shortpath[i,j] = min(shortpath[i,j], shortpath[i,k] + shortpath[k,j])
        return shortpath
    
    def shortestPaths(self, adjMatrices):
        shortestPaths = []
        for el in adjMatrices:
            shortestPaths.append(self.FloydWarshall(el))
        return shortestPaths


    def calcFrequencyVector(self, graph, delta):
        frequency = np.empty([delta+1, 1])
        for i in range(delta+1):
            frequency[i] = np.sum(graph == i)
        return frequency/norm(frequency)
      
    def deltaKernel(self, shortestPathGraph1, shortestPathGraph2):
        delta = int(np.maximum(np.max(shortestPathGraph1), np.max(shortestPathGraph2)))
        frequencyVector1 = self.calcFrequencyVector(shortestPathGraph1, delta)
        frequencyVector2 = self.calcFrequencyVector(shortestPathGraph2, delta)
        return  np.dot(np.transpose(frequencyVector1), frequencyVector2)[0]
    
    def getSimilarities(self, shortestPathgraphs):
        n = len(shortestPathgraphs)
        similarities = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                similarities[i,j] = self.deltaKernel(shortestPathgraphs[i], shortestPathgraphs[j])
        return similarities
