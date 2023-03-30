import pickle

import networkx as nx
import torch
import DataSetWave
from DataSetGen import *



#dataset = MyOwnDataset('DataGen/ret_lowret', sample = 2, delete = True)
#print(dataset[1])


filename = 'DataGen/ret_lowret/inst_' + str(1) + '.pickle'
with open(filename, 'rb') as f:
    [treeSet, retValues] = pickle.load(f)


net = nx.DiGraph()
for i in treeSet:
    net.add_nodes_from(treeSet[i].nodes())
    net.add_edges_from(treeSet[i].edges())

print(DataSetWave.getFlowEdgesUp(net))
print(DataSetWave.getFlowEdgesDown(net))

showGraph(net)