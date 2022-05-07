
from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np
from threading import Thread

def fit_n_components(D, Y, manifold_learning, n_neighbors = 14, n_iteration = 20):
    max_acc = 0.0
    max_idx = 1
    clf = svm.SVC(kernel="linear", C = 1.0)
    for i in range(2,n_iteration):
        ml_prj_D = manifold_learning(n_neighbors, i).fit_transform(D)
        scores_ln = cross_val_score(clf, ml_prj_D, Y, cv = 10, n_jobs= 8)
        if np.mean(scores_ln) > max_acc:
            max_acc = np.mean(scores_ln)
            max_idx = i
    return max_idx


import numpy as np
from numpy.linalg import norm

class FWExecutor(Thread):
    
    def __init__(self, graph_, shortest_path_method_, out_, idx_):
        Thread.__init__(self)
        self.graph = graph_
        self.shortest_path_method = shortest_path_method_
        self.out = out_
        self.idx = idx_
        
    def run(self):
        self.out[self.idx] = self.shortest_path_method(self.graph) 


class ShortestPathKernel():

    def initialize_paths(self, G):
        INF_ = float('inf')
        v = G.shape[0]
        dist = G
        dist[dist == 0] = INF_
        np.fill_diagonal(dist, 0)
        return dist
    
    def compute_FW_full(self, G):
        G = G.astype(np.float)
        dist = self.initialize_paths(G)
        v = G.shape[0]
        for k in range(v):
            for i in range(v):
                for j in range(v):
                    dist[i,j] = min(dist[i,j], dist[i,k] + dist[k,j])
        return dist
    
    def compute_FW(self, G):
        G = G.astype(np.float)
        dist = self.initialize_paths(G)
        v = G.shape[0]
        h = int(v/2)
        for k in range(v):
            for i in range(v):
                for j in range(i,v):
                    dist[i,j]= dist[j,i] = min(dist[i,j], dist[i,k] + dist[k,j])
        return dist
    
    def compute_shortest_paths(self, graphs):
        SP = []
        i = 0
        for adj_m in graphs:
            SP.append(self.compute_FW(adj_m))
            i+= 1
        return SP


    def compute_multi_shortest_paths(self, graphs):
        v = len(graphs)
        SP = np.empty((v,), dtype = np.object)
        THREAD_FOR_TIME = 6
  
        for i in range(0,v,THREAD_FOR_TIME):
            thr = []
            NTHREAD = np.minimum(v-i,THREAD_FOR_TIME)
            for j in range(NTHREAD):
                ex = FWExecutor(graphs[i+j],self.compute_FW, SP, i+j)
                thr.append(ex)
                ex.start()
            for j in range(NTHREAD):
                thr[j].join()
        return SP

    def extract_freq_vector(self, S, delta):
        F = np.empty([delta+1, 1])
        for i in range(delta+1):
            F[i] = np.sum(S == i)
        return F/norm(F)
    
    # similarity between frequency of paths
    def k_delta(self, SP1, SP2):
        delta = int(np.maximum(np.max(SP1), np.max(SP2)))

        F1 = self.extract_freq_vector(SP1, delta)
        F2 = self.extract_freq_vector(SP2, delta)
        return  np.dot(np.transpose(F1), F2)[0]#, F1, F2
    
            
    def kernel_similarity(self, SP1, SP2):
        return self.k_delta(SP1, SP2)
    
    def eval_similarities(self, SP_graphs):
        n = len(SP_graphs)
        K = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                K[i,j] = self.kernel_similarity(SP_graphs[i], SP_graphs[j])
        return K

    def threads_eval_similarities(self, graphs):
        n = len(graphs)
        K = np.zeros((n, n))
        THREAD_FOR_TIME = 2
        for i in range(0, n, THREAD_FOR_TIME):
            thr = []
            for j in range(THREAD_FOR_TIME):
                ex = KernelExecutor(self.kernel_similarity, K, graphs, i+j)
                thr.append(ex)
                ex.start()
                
            for j in range(THREAD_FOR_TIME):
                thr[j].join()
        return K
