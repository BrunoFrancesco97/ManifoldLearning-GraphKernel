import numpy as np
from grakel import Graph
import csv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV,cross_val_score
from sklearn.pipeline import make_pipeline
from grakel.kernels import ShortestPath
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict
from sklearn import manifold
from sklearn.model_selection import StratifiedKFold

def run(graphs,labels):
    
    strat_k_fold = StratifiedKFold(n_splits = 10, shuffle = True)
    # Values of C parameter of SVM
    C_grid = (10. ** np.arange(-4,6,1) / len(graphs)).tolist()

    # Creates pipeline
    estimator = make_pipeline(
        ShortestPath(algorithm_type="dijkstra",normalize=True,with_labels=False),
        GridSearchCV(SVC(kernel='precomputed'), dict(C=C_grid),
                    scoring='accuracy', cv=10))

    
    # Performs cross-validation and computes accuracy
    acc = accuracy_score(labels, cross_val_predict(estimator, graphs, labels, cv=strat_k_fold))


    n_neighbors = 4
    n_components = 18

    x_t = manifold.LocallyLinearEmbedding(n_neighbors=n_neighbors, n_components=n_components, n_jobs=-1).fit_transform()
    return acc
    