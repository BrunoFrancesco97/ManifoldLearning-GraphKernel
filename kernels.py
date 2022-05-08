import numpy as np
from sklearn import manifold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import pairwise_distances
import utils as utils
from shortestpath import ShortestPathKernel
from timeclass import *
import signal

clf = SVC(kernel="linear", C = 1.0)
kernel = ShortestPathKernel()
kFold = StratifiedKFold(n_splits = 10, shuffle = True) 


def timeout(signum, frame):   
    raise TimeoutException

signal.signal(signal.SIGALRM, timeout)

def run(numbers,label,folder):
    (graph,label) = utils.readFromFile(numbers,label,folder)
    adjMatrix = list()
    for el in graph:
        adjMatrix.append(el.get_adjacency_matrix())
    adjMatrix = np.array(adjMatrix,dtype=object)
    shortestPaths = kernel.shortestPaths(adjMatrix)
    similarities = kernel.getSimilarities(shortestPaths)
    distances = pairwise_distances(similarities, metric='euclidean',n_jobs=-1)
    scores_ln = cross_val_score(clf, distances, label, cv = kFold, n_jobs= -1)
    print("***************")
    print(folder)
    print("***************")
    print("Without Manifold Learning")
    utils.printResults(scores_ln,-1,-1)
    #manifoldLearning(clf,distances,label,kFold,0) #LLE
    manifoldLearning(clf,distances,label,kFold,1) #ISOMAP

def manifoldLearning(clf, D, label, kFold, flag): 
    best = None 
    worst = None
    i_b = 0
    j_b = 0
    i_w = 0
    j_w = 0
    for i in range(1,50):
        for j in range(1,50):
            if flag == 0:
                x_t = manifold.LocallyLinearEmbedding(n_neighbors=i,n_components=j).fit_transform(D)
            else:
                x_t = manifold.Isomap(n_neighbors=i,n_components=j).fit_transform(D)  
            try:
                signal.alarm(5)
                scores_new = cross_val_score(clf,x_t,label,cv=kFold,n_jobs=-1)
            except TimeoutException:
                signal.alarm(0)
                continue
            else:
                signal.alarm(0)
            if j == 1 and i == 1:
                best = scores_new
                worst = scores_new
                i_b = i
                j_b = j
                i_w = i
                j_w = j
            if np.mean(scores_new) < np.mean(worst):
                worst = scores_new
                i_b = i
                j_b = j
            if np.mean(scores_new) > np.mean(best):
                best = scores_new
                i_w = i
                j_w = j
    print("****")
    print("With Manifold Learning")
    if flag == 0:
        print("*** LLE ***")
    else: 
        print("*** ISOMAP ***")
    print("Worst: ")
    utils.printResults(worst,i_b,j_b)
    print("Best: ")
    utils.printResults(best,i_w,j_w)
