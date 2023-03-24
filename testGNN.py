import os
import pickle
import copy
from torch.nn import Linear
import torch
from torch_geometric.nn import GATv2Conv
from DataGen.pickClass import canPick, weightLeaf, reLabelTreeSet, pickCher, getLeaves
import torch.nn.functional as F
from DataSetPick import MyOwnDataset, toData2
import random
from Solvers2.TreeAn import showTree, pickRipe
from Solvers2.GreedyPicker import greedyPick
from heur.GNNheur import GNNheuristic
from heur.randomheur import randomheuristic



dataPickle = []


for I in range(4000,5000):

    #filename = 'DataGen/ret/inst_4000.pickle'
    filename = 'DataGen/ret/inst_' + str(I) + '.pickle'

    with open(filename, 'rb') as f:
        [treeSet, retValues] = pickle.load(f)
    toData2([copy.deepcopy(treeSet), retValues], type = 'tree')

    bestW = min([i[1] for i in retValues if i[1] != 0])
    bestPick = [i[0] for i in retValues if i[1] == bestW]




    ret = GNNheuristic(copy.deepcopy(treeSet), type = 'max')

    ret2 = 100
    for i in range(10):
        ret2 = min(GNNheuristic(copy.deepcopy(treeSet), type = 'random'), ret2)

    ret3 = 100
    for i in range(10):
        ret3 = min(randomheuristic(copy.deepcopy(treeSet)),ret2)

    print(ret,ret2,ret3,bestW)

    dataPickle.append([ret,ret2,ret3,bestW,I])

with open('testGNN.pickle', 'wb') as f:
    pickle.dump(dataPickle, f)