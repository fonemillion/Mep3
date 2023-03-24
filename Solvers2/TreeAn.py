import networkx as nx
import random
import copy
import matplotlib.pyplot as plt


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

def showGraph(Net, labels=True):
    subax1 = plt.plot()
    nx.draw_kamada_kawai(Net, with_labels=labels, font_weight='bold')
    # nx.draw_planar(Net, with_labels=True, font_weight='bold')
    plt.show()
def showGraphLabel(Net):
    subax1 = plt.plot()

    pos = nx.kamada_kawai_layout(Net)
    nx.draw(Net, pos)
    labels = nx.get_node_attributes(Net, 'children')
    #labels = nx.get_node_attributes(Net, 'inNrTrees')
    nx.draw_networkx_labels(Net, pos, labels)
    #nx.draw_kamada_kawai(Net, with_labels=labels, font_weight='bold')
    # nx.draw_planar(Net, with_labels=True, font_weight='bold')
    plt.show()
def getRoot(net):
    """
    returns root
    :param net: the network
    :return: list of roots
    """
    return [n for n in net.nodes() if net.in_degree(n) == 0][0]

def reLabelTreeSet(treeSet):
    leaves = getLeaves(treeSet[0])
    root = getRoot(treeSet[0])
    nodeDict = []
    nodeDict.append(allChildren(root,treeSet[0]))
    for leaf in leaves:
        nodeDict.append(allChildren(leaf,treeSet[0]))

    for tree in treeSet:
        for node in treeSet[tree]:
            check = allChildren(node,treeSet[tree])
            if check not in nodeDict:
                nodeDict.append(check)


    for tree in treeSet:
        mapping = dict()
        for node in treeSet[tree]:
            mapping[node] = nodeDict.index(allChildren(node,treeSet[tree]))
        treeSet[tree] = nx.relabel_nodes(treeSet[tree], mapping, copy=True)
    return nodeDict.copy()

def treeToNet(treeSet):
    """
    The treeSet it is returning is not labelled correct
    :param treeSet:
    :return:
    """
    treeSet = copy.deepcopy(treeSet)
    reLabelTreeSet(treeSet)
    net = copy.deepcopy(treeSet[0])
    # Define a dictionary of node labels
    labels = {}
    for node in net:
        labels[node] = allChildren(node,net)
    nx.set_node_attributes(net, labels, 'children')

    for i in range(1,len(treeSet)):
        treeSet[i]
        labels2 = {}
        for node in treeSet[i]:
            labels2[node] = allChildren(node,treeSet[i])
        nx.set_node_attributes(treeSet[i], labels2, 'children')
        #print(nx.get_node_attributes(tree2, 'children'))

        key_list = list(labels.keys())
        val_list = list(labels.values())
        #position = val_list.index(100)

        nrNode = len(net)
        mapping = {}
        for node in treeSet[i]:
            if labels2[node] in labels.values():
                position = val_list.index(labels2[node])
                newNode = key_list[position]
            else:
                newNode = nrNode
                nrNode += 1
            mapping[node] = newNode
        treeSet[i] = nx.relabel_nodes(treeSet[i], mapping)
        #print(mapping)
        net.add_nodes_from(treeSet[i].nodes(data=True))
        net.add_edges_from(treeSet[i].edges())

        #print(nx.get_node_attributes(treeSet[i], 'children'))
    inNrTrees = dict()
    for node in net:
        inNrTrees[node] = 0
    for i in treeSet:
        #showGraphLabel(treeSet[i])
        for node in treeSet[i]:
            inNrTrees[node] += 1
    nx.set_node_attributes(net, inNrTrees, 'inNrTrees')
    #showGraphLabel(net)


    return net,treeSet

def allChildren(node,net):
    """
    return list of all children of a node
    :param node:
    :param net:
    :return:
    """
    set = [node]
    notchild = []
    child = []
    while len(set) > 0:
        for i in set:
            for suc in net.successors(i):
                if net.out_degree(suc) == 0:
                    child.append(suc)
                else:
                    notchild.append(suc)
        set = notchild.copy()
        notchild = []
    if len(child) == 0:
        child.append(node)
    return sorted(child)


def pickRipe(treeSet):
    while True:
        ripe = ripeCherry(treeSet)
        if len(ripe) > 0:
            pickCher(treeSet, ripe[0][0])
            continue
        break

def ripeCherry(TSet):
    lvs = getLeaves(TSet[0])
    setall = [(l1, l2) for l1 in lvs for l2 in lvs if l1 != l2]
    set = [(l1, l2) for l1 in lvs for l2 in lvs if l1 != l2]
    for cher in setall: # doet dubbel voor (x,y) en (y,x)
        for i in TSet:
            if not isCher(TSet[i],lvs,cher):
                set.remove(cher)
                break
    return set

def getLeaves(net):
    return [u for u in net.nodes() if net.out_degree(u) == 0]

def isCher(tree,lSet,cher):
    node1 = cher[0]
    node2 = cher[1]
    if node1 not in lSet:
        return False
    if node2 not in lSet:
        return False
    par1 = list(tree.predecessors(node1))[0]
    par2 = list(tree.predecessors(node2))[0]
    return par1 == par2

def pickCher(treeSet,leaf):
    leaves = getLeaves(treeSet[0])
    for i in treeSet:
        if leaf in leaves:

            p = list(treeSet[i].predecessors(leaf))[0]
            if len(list(treeSet[i].predecessors(p))) == 0:
                # no grandparent
                treeSet[i].remove_node(leaf)
                treeSet[i].remove_node(p)
            else:
                gp = list(treeSet[i].predecessors(p))[0]
                chldren = list(treeSet[i].successors(p))
                chldren.remove(leaf)
                treeSet[i].remove_node(leaf)
                treeSet[i].remove_node(p)
                treeSet[i].add_edge(gp,chldren[0])

def labelTreeSet(treeSet):
    labels = {}


    for i in treeSet:
        for node in treeSet[i]:
            labels[node] = allChildren(node, treeSet[i])
        nx.set_node_attributes(treeSet[i], labels, 'children')
        # print(nx.get_node_attributes(tree2, 'children'))



def getSubSet(treeSet, leafSet):
    subTreeSet = dict()

    for tree in treeSet:
        toDo = leafSet.copy()
        newTree = nx.DiGraph()
        newTree.add_nodes_from(leafSet)
        while len(toDo) > 1:
            p = list(treeSet[tree].predecessors(toDo[0]))[0]
            set = [p]
            notleaf = []
            while len(set) > 0:
                for i in set:
                    for suc in treeSet[tree].successors(i):
                        if suc in toDo:
                            toDo.remove(suc)
                        else:
                            notleaf.append(suc)
                        newTree.add_edge(i,suc)
                set = notleaf.copy()
                notleaf = []
            toDo.append(p)

        #showGraph(newTree)
        subTreeSet[tree] = newTree.copy()

    return subTreeSet

