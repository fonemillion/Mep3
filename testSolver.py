import pickle
import time
import random
import copy

from Solvers.tempSolver import checkTemporal
from InstanceGenerators.netGen import genRandomTreeSet, net_to_tree2, genNetTC
from Solvers2.FPT3 import FPT3_2
from Solvers3.GreedyPicker3 import greedyPick3
from Solvers3.TempSolver import tempSolver
from Solvers3.TreeAn3 import removeTrivialLeaves, getLeaves
from Solvers3.TreeAn3 import showTreeSet

filename = 'test.pickle'

# Open the file in binary mode and pickle the object
with open(filename, 'rb') as f:
    treeSet = pickle.load(f)

print(tempSolver(treeSet))
print(FPT3_2(copy.deepcopy(treeSet)))
print(greedyPick3(copy.deepcopy(treeSet)))

showTreeSet(treeSet)