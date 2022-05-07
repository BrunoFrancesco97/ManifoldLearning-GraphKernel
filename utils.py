import csv
from grakel import Graph
import numpy as np 

def readFromFile(numbers,labelsFile,folder):
    files = list()
    graphs = list()
    labels = list()
    for el in numbers:
        string = "datasets/"+folder+"/graph"+el+".csv" 
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
    return (graphs,np.array(labels))