
from DataGen.pickClass import canPick, weightLeaf, reLabelTreeSet, pickCher, getLeaves
import random
from Solvers2.TreeAn import showTree, pickRipe


def randomheuristic(treeSet):

    ret = 0

    while True:

        pickRipe(treeSet)
        if len(getLeaves(treeSet[0])) < 3:
            # print("no more leaves" , ret)
            break
        # for i in treeSet:
        #    showTree(treeSet[i])

        reLabelTreeSet(treeSet)
        options = canPick(treeSet)
        if len(options) == 0:
            # print("no options")
            ret = 100
            break
        if len(options) == 1:
            # print("only 1 option")
            ret += weightLeaf(treeSet, options[0])
            pickCher(treeSet, options[0])

        else:
            pickSet = canPick(treeSet)
            pickSet.sort()
            leaf = random.choice(pickSet)
            # print("multiple choise ", leaf, pickSet, weight)
            ret += weightLeaf(treeSet, leaf)
            pickCher(treeSet, leaf)

    return ret