import numpy as np
from sklearn import manifold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import pairwise_distances
import utils as utils
from shortestpath import ShortestPathKernel
import signal

clf = SVC(kernel="linear", C = 1.0)
kernel = ShortestPathKernel()
kFold = StratifiedKFold(n_splits = 10, shuffle = True) 

class TimeoutException(Exception):  
    pass

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
    res = utils.prepareResults(scores_ln,-1,-1,"\n***************"+folder+"***************\nWithout Manifold Learning\n")
    res = res + manifoldLearning(clf,distances,label,kFold,0) #LLE
    res = res + manifoldLearning(clf,distances,label,kFold,1) #ISOMAP
    utils.writeToFile(res)
    print("** END! CHECK THE RESULT FILE! **")

def manifoldLearning(clf, D, label, kFold, flag): 
    best = None 
    worst = None
    i_b = 0
    j_b = 0
    i_w = 0
    j_w = 0
    res = ""
    for i in range(2,31):
        for j in range(2,21):
            if flag == 0:
                x_t = manifold.LocallyLinearEmbedding(n_neighbors=i,n_components=j).fit_transform(D)
            else:
                x_t = manifold.Isomap(n_neighbors=i,n_components=j).fit_transform(D)  
            try:
                signal.alarm(10)
                scores_new = cross_val_score(clf,x_t,label,cv=kFold,n_jobs=-1)
            except TimeoutException:
                signal.alarm(0)
                continue
            else:
                signal.alarm(0)
            if j == 2 and i == 2:
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
    if flag == 0:
        res = res + utils.prepareResults(best,i_b,j_b,"\nWith Manifold Learning\n*** LLE ***\nBest: ")
        res = res + utils.prepareResults(worst,i_w,j_w,"\nWorst: ")
    else: 
        res = res + utils.prepareResults(best,i_b,j_b,"\nWith Manifold Learning\n*** ISOMAP ***\nBest: ")
        res = res + utils.prepareResults(worst,i_w,j_w,"\nWorst: ")
    return res 
