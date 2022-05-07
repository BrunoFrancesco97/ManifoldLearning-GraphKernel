

import numpy as np
from sklearn import manifold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import pairwise_distances
import utils as utils
from shortestpath import ShortestPathKernel

def run(numbers,label,folder):
    (graph,label) = utils.readFromFile(numbers,label,folder)
    adjMatrix = list()
    for el in graph:
        adjMatrix.append(el.get_adjacency_matrix())
    adjMatrix = np.array(adjMatrix,dtype=object)

    sp_kernel = ShortestPathKernel()

    SP_graphs = sp_kernel.shortest_paths(adjMatrix)

    K = sp_kernel.get_similarities(SP_graphs)

    D = pairwise_distances(K, metric='euclidean',n_jobs=-1)

    strat_k_fold = StratifiedKFold(n_splits = 10, shuffle = True) 

    clf = SVC(kernel="linear", C = 1.0)

    scores_ln = cross_val_score(clf, D, label, cv = strat_k_fold, n_jobs= -1)
    print("***************")
    print(folder)
    print("***************")
    print("Without Manifold Learning")
    utils.printResults(scores_ln,-1,-1)
    best = None 
    worst = None
    i_b = 0
    j_b = 0
    i_w = 0
    j_w = 0
    for i in range(1,50):
        for j in range(1,50):
            x_t = manifold.LocallyLinearEmbedding(n_neighbors=i,n_components=j).fit_transform(D)
            clf = SVC(kernel="linear", C = 1.0)
            scores_new = cross_val_score(clf,x_t,label,cv=10,n_jobs=-1)
            if j == 1 and i == 1:
                best = scores_new
                worst = scores_new
                i_b = i
                j_b = j
                i_w = i
                j_w = j
            if np.mean(scores_new) < np.mean(worst):
                worst = scores_new
                i_w = i
                j_w = j
            if np.mean(scores_new) > np.mean(best):
                best = scores_new
                i_b = i
                j_b = j
    print("****")
    print("With Manifold Learning")
    print("Worst: ")
    utils.printResults(worst,i_w,j_w)
    print("Best: ")
    utils.printResults(best,i_b,j_b)