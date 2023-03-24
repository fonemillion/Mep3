import copy
import os
import pickle
import random

from matplotlib import pyplot as plt

from InstanceGenerators.netGen import genRandomTreeSet, net_to_tree2, genNetTC
from Solvers.tempSolver import checkTemporal
from Solvers2.FPT3 import FPT3_2, FPT3
from Solvers2.GreedyPicker import greedyPick
from Solvers2.TreeAn import pickRipe
from DataGen.pickClass import *


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

def showTree(tree,labels=True):
    subax1 = plt.plot()

    root = [u for u in tree.nodes() if tree.in_degree(u) == 0]
    root = root[0]
    pos = hierarchy_pos(tree, root)
    nx.draw(tree, pos=pos, with_labels=True, font_weight='bold')
    plt.show()
def showTreeSet(treeSet,labels=True):
    for i in treeSet:
        plt.figure(i)
        subax1 = plt.plot()
        root = [u for u in treeSet[i].nodes() if treeSet[i].in_degree(u) == 0]
        root = root[0]
        pos = hierarchy_pos(treeSet[i], root)
        nx.draw(treeSet[i], pos=pos, with_labels=True, font_weight='bold')
    plt.show()



while True:

    if random.random() > 0.5:
        treeSet = genRandomTreeSet(20, 2)
    else:
        treeSet = net_to_tree2(genNetTC(20,10),2)

    while True:
        pickRipe(treeSet)
        options = canPick(treeSet)
        if len(options) == 1:
            pickCher(treeSet,options[0])
            continue
        break
    reLabelTreeSet(treeSet)
    if not checkTemporal(treeSet):
        print("non-temporal")
        showTreeSet(treeSet)

        continue


