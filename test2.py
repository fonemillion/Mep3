import pickle
import copy
from Solvers2.FPT3 import FPT3_2
from Solvers2.GreedyPicker import greedyPick
from Solvers2.TreeAn import showTree, pickRipe
from DataGen.pickClass import *


filename = 'test.pickle'
with open(filename, 'rb') as f:
    treeSet = pickle.load(f)
pickRipe(treeSet)
reLabelTreeSet(treeSet)

pickCher(treeSet,11)
reLabelTreeSet(treeSet)
pickCher(treeSet,9)
reLabelTreeSet(treeSet)
pickCher(treeSet,4)
reLabelTreeSet(treeSet)
pickCher(treeSet,4)
pickCher(treeSet,6)
pickCher(treeSet,8)
reLabelTreeSet(treeSet)
print(FPT3_2(copy.deepcopy(treeSet)))
print(greedyPick(copy.deepcopy(treeSet)))
for i in treeSet:
    showTree(treeSet[i])

