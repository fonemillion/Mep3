import copy
import os
import pickle
import random
from InstanceGenerators.netGenTemp import net_to_tree2, simulation_temp
#from Solvers.tempSolver import checkTemporal
#from Solvers2.FPT3 import FPT3_2, FPT3
#from Solvers2.GreedyPicker import greedyPick
#from Solvers2.TreeAn import pickRipe
from DataGen.pickClass import *
from Solvers3.GreedyPicker3 import greedyPick3
from Solvers3.TempSolver import tempSolver
from Solvers3.TreeAn3 import removeTrivialLeaves, showTreeSet

i = 0
j = 0
while i < 5000:

    filename = "ret/inst_" + str(i) + ".pickle"

    if os.path.exists(filename):
        print(f"The file {filename} already exists.")
        i += 1
        continue


    R = random.randint(2,5)
    L = 3 + R + random.randrange(16)

    net, _ = simulation_temp(L, R)
    treeSet = net_to_tree2(net,2)

    if len(tempSolver(treeSet)) == 0:
        print("non-temporal")
        break

    removeTrivialLeaves(treeSet)
    reLabelTreeSet(treeSet)
    #showTreeSet(treeSet)
    if len(treeSet[0]) < 2:
        continue

    options = canPick(treeSet)
    if len(options) > 1:
        retValues = []
        for leaf in options:
            subTreeSet = copy.deepcopy(treeSet)
            sol1 = weightLeaf(treeSet, leaf)
            pickCher(subTreeSet, leaf)
            sol = greedyPick3(subTreeSet)
            #print(sol)
            if len(sol) == 0:
                sol1 = -1
            else:
                sol1 += sol[0][0]
            retValues.append([leaf,sol1])
            del sol
        retV = {r[1] for r in retValues}
        if len(retV) == 1:
            print('only good solutions')
            continue
        with open(filename, 'wb') as f:
            pickle.dump([treeSet,retValues], f)
        print(i,R,retV,len(retValues))
        i += 1
        continue


    else:
        print("Only 1 pickable ")
        pickCher(treeSet,options[0])

