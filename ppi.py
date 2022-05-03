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
numbers = ['001','002','003','004','005','006','007','008','009','010','011','012','013','014','015','016','017','018','019','020','021','022','023','024','025','026','027','028','029','030','031','032','033','034','035','036','037','038','039','040','041','042','043','044','045','046','047','048','049','050','051','052','053','054','055','056','057','058','059','060','061','062','063','064','065','066','067','068','069','070','071','072','073','074','075','076','077','078','079','080','081','082','083','084','085','086']
labelsFile = "datasets/PPI/labels.csv"
folder = "PPI"

def run():
    files = list()
    graphs = list()
    labels = list()
    for el in numbers:
        string = "datasets/PPI/graph"+el+".csv" 
        files.append(string)
    for el in files:
        file = open(el, "r")
        csvreader = csv.reader(file)
        rows = []
        for row in csvreader:
                rows.append(row)

        matrix_size = len(rows)
        edges = list()
        for i in range(0,matrix_size):
            for j in range(0,matrix_size):
                if(rows[i][j] == "1"):
                    edges.append((i,j))
        graphs.append(Graph(edges))

    file = open(labelsFile, "r")
    csvreader = csv.reader(file)
    for el in csvreader:
        labels.append(int(el[0]))

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
    print(acc)