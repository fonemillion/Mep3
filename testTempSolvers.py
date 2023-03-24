
import pickle
import time
import random
import copy

from Solvers.tempSolver import checkTemporal
from InstanceGenerators.netGen import genRandomTreeSet, net_to_tree2, genNetTC
from Solvers3.TempSolver import tempSolver
from Solvers3.TreeAn3 import removeTrivialLeaves, getLeaves, reLabelTreeSet

time1 = []
time2 = []
ret = []
i = 0

while True:


    if random.random() > 0.9:
        treeSet = genRandomTreeSet(random.randint(6+ int(i/100),10 + int(i/100)), 2)
    else:
        treeSet = net_to_tree2(genNetTC(random.randint(10+ int(i/100), 20+ int(i/100)),random.randint(3, 8+ int(i/100))),random.randint(2,3 + int(i/100)))
    reLabelTreeSet(treeSet)
    removeTrivialLeaves(treeSet)
    if len(getLeaves(treeSet[0])) < 6:
        continue

    start_time = time.perf_counter()
    check1 = checkTemporal(copy.deepcopy(treeSet))
    elapsed_time1 = time.perf_counter() - start_time
    print(check1)
    print("Elapsed time for function 2: ", elapsed_time1)
    time1.append(elapsed_time1)

    start_time = time.perf_counter()
    sol = tempSolver(copy.deepcopy(treeSet))
    check2 = len(sol) != 0
    print(check2)
    elapsed_time2 = time.perf_counter() - start_time
    print("Elapsed time for function 2: ", elapsed_time2)

    time2.append(elapsed_time2)
    print('------------------------------------------------------')

    if check1 != check2:
        filename = 'test.pickle'

        # Open the file in binary mode and pickle the object
        with open(filename, 'wb') as f:
            pickle.dump(treeSet, f)
        break

    filename = 'time.pickle'
    ret.append(check1)
    # Open the file in binary mode and pickle the object
    with open(filename, 'wb') as f:
        pickle.dump([time1,time2,ret], f)
        #print(ret)
    i += 1

