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

def prepareResults(scores,neighbors,dim,labels):
    res = labels
    res = res + "\nMin: "+str(np.min(scores))
    res = res + "\nMean:"+str(np.mean(scores))
    res = res + "\nMax: "+str(np.max(scores))
    if neighbors != 0:
        res = res + "\nNeighbors: "+str(neighbors)
    if dim != 0:
        res = res + "\nDimension: "+str(dim)
    return res 
    
def writeToFile(stringToWrite):
    f = open("results.txt", "a")
    f.write(stringToWrite)
    f.close()

def printResults(scores,neighbors,dim):
    print("Min: "+str(np.min(scores)))
    print("Mean:"+str(np.mean(scores)))
    print("Max: "+str(np.max(scores)))
    if neighbors != 0:
        print("Neighbors: "+str(neighbors))
    if dim != 0:
        print("Dimension: "+str(dim))
