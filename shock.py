import numpy as np
from grakel import Graph
import csv
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from grakel.kernels import ShortestPath
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict

def run(graphs, labels):
    graph_train, graph_test, labels_train, labels_test = train_test_split(graphs, labels, test_size=0.1)
    # Uses the shortest path kernel to generate the kernel matrices
    gk = ShortestPath(algorithm_type="dijkstra",normalize=True, with_labels=False)
    K_train = gk.fit_transform(graph_train)
    K_test = gk.transform(graph_test)

    # Uses the SVM classifier to perform classification
    clf = SVC(kernel="precomputed")
    clf.fit(K_train, labels_train)
    y_pred = clf.predict(K_test)

    acc = accuracy_score(labels_test, y_pred)
    return acc