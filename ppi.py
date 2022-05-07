

import scipy.io as sio
import numpy as np
from sklearn import manifold
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.decomposition import PCA
import utils as utils
from shortestpath import ShortestPathKernel, fit_n_components

def run():
    numbersPPI = ['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','024','025','026','027','028','029','030','031','032','033','034','035','036','037','038','039','040','041','042','043','044','045','046','047','048','049','050','051','052','053','054','055','056','057','058','059','060','061','062','063','064','065','066','067','068','069','070','071','072','073','074','075','076','077','078','079','080','081','082','083','084','085','086']
    labelsFilePPI = "datasets/PPI/labels.csv"
    folderPPI = "PPI"
    (graphPPI,labelsPPI) = utils.readFromFile(numbersPPI,labelsFilePPI,folderPPI)
    adjMatrix = list()
    for el in graphPPI:
        adjMatrix.append(el.get_adjacency_matrix())
    adjMatrix = np.array(adjMatrix,dtype=object)

    sp_kernel = ShortestPathKernel()

    SP_graphs = sp_kernel.compute_multi_shortest_paths(adjMatrix)

    K = sp_kernel.eval_similarities(SP_graphs)

    D = pairwise_distances(K, metric='euclidean',n_jobs=4)

    strat_k_fold = StratifiedKFold(n_splits = 10, shuffle = True) 

    clf = SVC(kernel="linear", C = 1.0)

    scores_ln = cross_val_score(clf, D, labelsPPI, cv = strat_k_fold, n_jobs= -1)
    print(str(np.min(scores_ln)) +" - "+str(np.mean(scores_ln))+ " - " + str(np.max(scores_ln)) + " - "+ str(np.std(scores_ln)))
