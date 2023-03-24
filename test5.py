
import pickle
import time
import random
import networkx as nx
import matplotlib.pyplot as plt
import copy

from Solvers2.FPT2 import FPT2
from Solvers2.FPT3 import FPT3, FPT3_2
from Solvers.tempSolver import checkTemporal
from InstanceGenerators.netGen import genRandomTreeSet, net_to_tree2, genNetTC
from Solvers2.TreeAn import pickRipe, reLabelTreeSet
from Solvers2.GreedyPicker import greedyPick, Case, treeToNet,getRoot



def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    '''
    From Joel's answer at https://stackoverflow.com/a/29597209/2966723.
    Licensed under Creative Commons Attribution-Share Alike

    If the graph is a tree this will return the positions to plot this in a
    hierarchical layout.

    G: the graph (must be a tree)

    root: the root node of current branch
    - if the tree is directed and this is not given,
      the root will be found and used
    - if the tree is directed and this is given, then
      the positions will be just for the descendants of this node.
    - if the tree is undirected and not given,
      then a random choice will be used.

    width: horizontal space allocated for this branch - avoids overlap with other branches

    vert_gap: gap between levels of hierarchy

    vert_loc: vertical location of root

    xcenter: horizontal location of root
    '''
    if not nx.is_tree(G):
        raise TypeError('cannot use hierarchy_pos on a graph that is not a tree')

    if root is None:
        if isinstance(G, nx.DiGraph):
            root = next(iter(nx.topological_sort(G)))  # allows back compatibility with nx version 1.11
        else:
            root = random.choice(list(G.nodes))

    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, pos=None, parent=None):
        '''
        see hierarchy_pos docstring for most arguments

        pos: a dict saying where all nodes go if they have been assigned
        parent: parent of this branch. - only affects it if non-directed

        '''

        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if not isinstance(G, nx.DiGraph) and parent is not None:
            children.remove(parent)
        if len(children) != 0:
            dx = width / len(children)
            nextx = xcenter - width / 2 - dx / 2
            for child in children:
                nextx += dx
                pos = _hierarchy_pos(G, child, width=dx, vert_gap=vert_gap,
                                     vert_loc=vert_loc - vert_gap, xcenter=nextx,
                                     pos=pos, parent=root)
        return pos

    return _hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)


def showGraph(Net,labels=True):
    subax1 = plt.plot()
    nx.draw_kamada_kawai(Net, with_labels=labels, font_weight='bold')
    #nx.draw_planar(Net, with_labels=True, font_weight='bold')
    plt.show()

def showTree(tree,labels=True):
    subax1 = plt.plot()
    root = [u for u in tree.nodes() if tree.in_degree(u) == 0]
    root = root[0]
    pos = hierarchy_pos(tree, root)
    nx.draw(tree, pos=pos, with_labels=True, font_weight='bold')
    plt.show()


time1 = []
time2 = []

while True:
    if random.random() > 0.8:
        treeSet = genRandomTreeSet(10,2)
    else:
        treeSet = net_to_tree2(genNetTC(16,random.randint(2,5)),4)

    pickRipe(treeSet)
    reLabelTreeSet(treeSet)
    if not checkTemporal(treeSet):
        print("non-temporal")
        continue



    sol1 = FPT3(copy.deepcopy(treeSet))
    if len(sol1) == 0:
        sol1 = [[10]]
    print(sol1)



    net, inNrTrees = treeToNet(treeSet)
    case = Case(net, inNrTrees)

    root = getRoot(case.net)
    weight = 0
    lvs = []
    for cher in case.cherries:
        children = list(case.net.successors(cher))
        if children[0] in lvs or children[1] in lvs:
            continue
        cherSplit = False
        for node in case.net.successors(root[0]):
            subLeaves = case.childD[case.childK.index(node)]
            if children[0] in subLeaves and children[1] not in subLeaves:
                #print(children[0],children[1],subLeaves)
                cherSplit = True
        if cherSplit:
            weight += 1
            #print(children[0],children[1])
            lvs.append(children[0])
            lvs.append(children[1])




    print("w",weight)

    if sol1[0][0]  < weight:
        for i in treeSet:
            showTree(treeSet[i])
        break




