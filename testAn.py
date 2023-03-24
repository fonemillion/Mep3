from InstanceGenerators.netGen import genRandomTreeSet
from Solvers3.TreeAn3 import *
from Solvers3.TempSolver import tempSolver


treeSet = genRandomTreeSet(20,2)
reLabelTreeSet(treeSet)
print(tempSolver(treeSet))
showTreeSet(treeSet)
