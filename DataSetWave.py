import copy
import pickle
import os
import shutil

import networkx
import networkx as nx
from torch_geometric.data import Data
import torch
from torch_geometric.data import InMemoryDataset, download_url
from DataGen.pickClass import reLabelTreeSet, weightLeaf, allChildren,canPick, getLeaves, getRoot
from Solvers2.TreeAn import showGraph

class MyOwnDataset(InMemoryDataset):
    def __init__(self, name, transform=None, pre_transform=None, pre_filter=None, sample = None, delete = False):
        dir = name + 'data'


        if delete == True:
            if os.path.exists(dir):
                shutil.rmtree(dir)
                print(f"{dir} deleted successfully.")
            else:
                print(f"{dir} does not exist.")

        self.sample = sample
        self.name = name
        super().__init__(dir, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])


#    @property
#    def raw_file_names(self):
#        return ['some_file_1', 'some_file_2', ...]

    @property
    def processed_file_names(self):
        return ['data.pt']

#    def download(self):
#        # Download to `self.raw_dir`.
#        #download_url(url, self.raw_dir)
#        ...

    def process(self):
        # Read data into huge `Data` list.
        data_list = []

        # read sample size
        if self.sample == None:
            N = 200
        elif self.sample == 'all':
            N = 1000000000
        else:
            N = self.sample
        i = 0
        while True:
            filename = self.name + '/inst_' + str(i) + '.pickle'
            i += 1
            if not os.path.isfile(filename):
                break
            if len(data_list) >= N:
                break

            data = toData(filename)
            if data == None:
                raise Exception("no Data")

            # no trivial cases
            with open(filename, 'rb') as f:
                [treeSet, retValues] = pickle.load(f)
            del treeSet
            if len({j[1] for j in retValues}) == 1:
                continue

            data_list.append(data)




        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def getFlowEdgesUp(net):
    edgeDict = dict()
    lvs = getLeaves(net)
    done = lvs.copy()
    i = 0
    while len(done) < len(net.nodes):

        nodes = [n for n in net.nodes if n not in done]
        toDo = []
        # Find nodes
        for node in nodes:
            if all([n in done for n in net.successors(node)]):
                toDo.append(node)
        # create edges
        edges = []
        for item in toDo:
            done.append(item)
            for child in net.successors(item):
                edges.append((child,item))
        edgeDict[i] = edges
        i += 1
    return edgeDict


def getFlowEdgesDown(net):
    root = getRoot(net)
    edgeDict = dict()
    done = [root]
    i = 0
    while len(done) < len(net.nodes):

        nodes = [n for n in net.nodes if n not in done]
        toDo = []
        # Find nodes
        for node in nodes:
            if all([n in done for n in net.predecessors(node)]):
                toDo.append(node)
        # create edges
        edges = []
        for item in toDo:
            done.append(item)
            for child in net.predecessors(item):
                edges.append((child, item))
        edgeDict[i] = edges
        i += 1
    return edgeDict

def toData(filename):
    # get Data
    with open(filename, 'rb') as f:
        [treeSet, retValues] = pickle.load(f)

    # test is labeled
    nodeDict = getLabelTreeSet(treeSet)
    if nodeDict == None:
        raise Exception('error treeSet not correctly labbeled')

    net = nx.DiGraph()
    for i in treeSet:
        net.add_nodes_from(treeSet[i].nodes())
        net.add_edges_from(treeSet[i].edges())

    lvs = getLeaves(net)


    pickAble = canPick(treeSet)
    isLeaf = []
    #xDict = []
    zDict = []
    pickset = []
    for node in net:
        zDict.append(node)
        if node in lvs:
            #xDict.append(node)
            isLeaf.append(True)
        else:
            isLeaf.append(False)

        pickset.append(node in pickAble)



    nonTemp = []
    for leaf in retValues:
        if leaf[1] == -1:
            nonTemp.append(leaf[0])

    retValues = [i for i in retValues if i[0] not in nonTemp]

    bestW = min([i[1] for i in retValues])
    bestPick = [i[0] for i in retValues if i[1] == bestW]




    x = []
    y = []
    root = getRoot(net)

    #print(lvs)

    H = net.to_undirected()
    leaves = [u for u in net.nodes() if net.out_degree(u) == 0]
    root = getRoot(net)
    cycles = nx.cycle_basis(H, root)
    # del H

    bridgeNodes = []

    for bridge in nx.bridges(H, root=None):
        if bridge[0] not in bridgeNodes:
            bridgeNodes.append(bridge[0])
        if bridge[1] not in bridgeNodes:
            bridgeNodes.append(bridge[1])

    z = []
    for node in zDict:
        nr = 0
        type = 0
        for cyc in cycles:
            if node in cyc:
                nr += 1
        if node in leaves:
            type = 1
        if node == root:
            type = -1
        inBridge = 0
        if node in bridgeNodes:
            inBridge = 1
        # list1.append([net.out_degree(node), net.in_degree(node),net.out_degree(node) + net.in_degree(node),len(allChildren(node,net)),net.number_of_nodes(),nr])
        #z.append([len(nodeDict[node]), type])
        nrChild = len(nodeDict[node])

        rootLmax = 0
        rootLmin = len(lvs)
        depthLmax = 0
        depthLmin = len(lvs)
        depthSmax = 0
        depthSmin = len(lvs)
        for tree in treeSet:
            try:
                rootL = findLengthRoot(treeSet[tree],node)
                rootLmax = max(rootLmax,rootL)
                rootLmin = min(rootLmin,rootL)
            except:
                None

            depthL = 0
            depthS = len(lvs)
            for target in lvs:
                try:
                    depthL = max(depthL,nx.shortest_path_length(treeSet[tree], source=node, target=target))
                    depthS = min(depthS,nx.shortest_path_length(treeSet[tree], source=node, target=target))
                except:
                    None
            depthLmax = max(depthLmax,depthL)
            depthLmin = min(depthLmin,depthL)
            depthSmax = max(depthSmax,depthS)
            depthSmin = min(depthSmin,depthS)


        z.append([net.out_degree(node), net.in_degree(node), len(nodeDict[node]), nr, type, inBridge, nrChild, rootLmax, rootLmin, depthLmax, depthLmin, depthSmax, depthSmin])

    #print(net.edges)

    z1Edges = getFlowEdges()


    for edge in net.edges:
        z1Edges.append((zDict.index(edge[0]),zDict.index(edge[1])))
        z1Edges.append((zDict.index(edge[1]),zDict.index(edge[0])))




    # Convert the graph to a TorchGeometric Data object
    data = Data()


    #data.nodeDict = nodeDict
    data.zDict = zDict
    data.x = torch.tensor(x, dtype=torch.float)
    data.z = torch.tensor(z, dtype=torch.float)
    data.edge_index = torch.tensor(edge_index).t().contiguous()
    data.z1edge_index = torch.tensor(z1Edges).t().contiguous()
    data.z2edge_index = torch.tensor(z2Edges).t().contiguous()
    data.z3edge_index = torch.tensor(z3Edges).t().contiguous()
    data.z4edge_index = torch.tensor(z4Edges).t().contiguous()
    data.z5edge_index = torch.tensor(z5Edges).t().contiguous()
    data.edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    data.y = torch.tensor(y, dtype=torch.long)
    data.pickable = torch.tensor(pickset, dtype=torch.bool)
    data.isLeaf = torch.tensor(isLeaf, dtype=torch.bool)


    del net, nodeDict
    return data


def findLengthRoot(tree, leaf):
    root = getRoot(tree)
    lvs = getLeaves(tree)
    if leaf == root:
        return 0
    if leaf not in tree:
        raise Exception( leaf, " not in network")
    p = list(tree.predecessors(leaf))[0]
    length = 1
    while p != root:
        p = list(tree.predecessors(p))[0]
        length += 1
    return length

def findLengthTo(tree, leaf, node):
    lvs = getLeaves(tree)
    if leaf not in lvs:
        raise Exception( leaf, " not in network")
    p = list(tree.predecessors(leaf))[0]
    length = 1
    while p != node:
        p = list(tree.predecessors(p))[0]
        length += 1
    return length

def getLeafFeat(treeSet,node,nodeDict):
    lvs = getLeaves(treeSet[0])
    min_to_root = len(lvs) + 2
    max_to_root = 0
    for tree in treeSet:
        length = findLengthRoot(treeSet[tree], node)
        min_to_root = min(min_to_root, length)
        max_to_root = max(max_to_root, length)

    listP = getPred(treeSet, node)
    #print(listP)
    listPND = dict()
    for j in treeSet:
        #listP[j].remove(0)
        listPND[j] = [nodeDict[i] for i in listP[j]]



    cycDistMax = 0
    cycDistMin = len(lvs) ** 2
    cycDistMax2 = 0
    cycDistMin2 = len(lvs) ** 2
    #cycDistMax3 = 0
    #cycDistMin3 = len(lvs) ** 2
    loop = [(i, j) for i in treeSet for j in treeSet if i != j and i < j] # needs to be cleaned
    for item in loop:
        i = item[0]
        j = item[1]
        seti = copy.deepcopy(listPND[i])
        setj = copy.deepcopy(listPND[j])
        dist = 0
        #for z in range(1,len(lvs)):
        #    for leaf in lvs:
        #        dist += abs(int(leaf in seti[0]) * z/len(seti[0]) - int(leaf in setj[0]) * z/len(setj[0]))
        #    if z >= len(seti[0]):
        #        seti.pop(0)
        #    if z >= len(setj[0]):
        #        setj.pop(0)
        for I in seti:
            smallest_dist = min([len(set(I) ^ set(J)) for J in setj])
            dist += smallest_dist
        for J in setj:
            smallest_dist = min([len(set(I) ^ set(J)) for I in seti])
            dist += smallest_dist

        cycDistMax = max(dist,cycDistMax)
        cycDistMin = min(dist,cycDistMin)

        #cycDistMax2 = max(len(set(listPND[i][0]) - set(listPND[j][0])), cycDistMax2)
        #cycDistMin2 = min(len(set(listPND[i][0]) - set(listPND[j][0])), cycDistMin2)
        cycDistMax2 = max(len(set(listPND[i][0]) ^ set(listPND[j][0])), cycDistMax2)
        cycDistMin2 = min(len(set(listPND[i][0]) ^ set(listPND[j][0])), cycDistMin2)
        #else:
        #    cycDistMin = 0
        #    cycDistMin2 = 0

    #print(node, cycDistMax/len(lvs), cycDistMin/len(lvs))
    #print(listPND)




    return [min_to_root,max_to_root,cycDistMax,cycDistMin,cycDistMax2,cycDistMin2]

def getEdgeFeat(nodeDict,treeSet,pair):
    lvs = getLeaves(treeSet[0])
    root = getRoot(treeSet[0])
    max_nr_pick_before_leaf = 0
    min_nr_pick_before_leaf = len(lvs)

    max_length_same_par_l1 = 0
    min_length_same_par_l1 = len(lvs)

    max_length_same_par_l2 = 0
    min_length_same_par_l2 = len(lvs)
    isCherry = 0
    blocked = 0
    for tree in treeSet:
        sameP = []
        for node in treeSet[tree]:
            if pair[0] in nodeDict[node] and pair[1] in nodeDict[node]:
                sameP.append(node)
        sameP = sameP[argmin([len(nodeDict[n]) for n in sameP])]
        #print(sameP)
        nr_pick_before_leaf = len(nodeDict[sameP])-2
        length_same_par_l1 = findLengthTo(treeSet[tree],pair[0],sameP)
        length_same_par_l2 = findLengthTo(treeSet[tree],pair[1],sameP)

        max_nr_pick_before_leaf = max(max_nr_pick_before_leaf,nr_pick_before_leaf)
        min_nr_pick_before_leaf = min(min_nr_pick_before_leaf,nr_pick_before_leaf)

        max_length_same_par_l1 = max(max_length_same_par_l1,length_same_par_l1)
        min_length_same_par_l1 = min(min_length_same_par_l1,length_same_par_l1)

        max_length_same_par_l2 = max(max_length_same_par_l2,length_same_par_l2)
        min_length_same_par_l2 = min(min_length_same_par_l2,length_same_par_l2)
        if list(treeSet[tree].predecessors(pair[0])) == list(treeSet[tree].predecessors(pair[1])):
            isCherry = 1
        childRoot = list(treeSet[tree].successors(root))
        if (pair[0] in nodeDict[childRoot[0]] and pair[1] in nodeDict[childRoot[1]]) or (pair[1] in nodeDict[childRoot[0]] and pair[0] in nodeDict[childRoot[1]]):
            blocked += 1
    #print(nodeDict)
    distPPmin = len(lvs) **2
    distPPmax = 0
    cycDistMax = 0
    cycDistMin = len(lvs) ** 2

    loop = [(i, j) for i in treeSet for j in treeSet if i != j and i < j]
    for item in loop:
        i = item[0]
        j = item[1]
        samePi = []
        for node in treeSet[i]:
            if pair[0] in nodeDict[node] and pair[1] in nodeDict[node]:
                samePi.append(node)
        samePi = samePi[argmin([len(nodeDict[n]) for n in samePi])]

        samePj = []
        for node in treeSet[j]:
            if pair[0] in nodeDict[node] and pair[1] in nodeDict[node]:
                samePj.append(node)
        samePj = samePj[argmin([len(nodeDict[n]) for n in samePj])]
        #print(samePi,samePj)
        pathi0 = [pair[0]]
        pathi1 = [pair[1]]
        pathj0 = [pair[0]]
        pathj1 = [pair[1]]
        p = pair[0]
        while True:
            p = list(treeSet[i].predecessors(p))[0]
            pathi0.append(p)
            if p == samePi:
                break
        p = pair[1]
        while True:
            p = list(treeSet[i].predecessors(p))[0]
            if p == samePi:
                break
            pathi1.append(p)
        p = pair[0]
        while True:
            p = list(treeSet[j].predecessors(p))[0]
            pathj0.append(p)
            if p == samePj:
                break
        p = pair[1]
        while True:
            p = list(treeSet[j].predecessors(p))[0]
            if p == samePj:
                break
            pathj1.append(p)
        pathi1.reverse()
        pathj1.reverse()
        pathi = pathi0 + pathi1
        pathj = pathj0 + pathj1
        pathi = [nodeDict[i] for i in pathi]
        pathj = [nodeDict[i] for i in pathj]
        distPP = len(set(nodeDict[samePi]) ^ set(nodeDict[samePj]))
        distPPmin = min(distPPmin,distPP)
        distPPmax = max(distPPmax,distPP)

        dist = 0
        for I in pathi:
            smallest_dist = min([len(set(I) ^ set(J)) for J in pathj])
            dist += smallest_dist
        for J in pathj:
            smallest_dist = min([len(set(I) ^ set(J)) for I in pathi])
            dist += smallest_dist
        cycDistMax = max(dist,cycDistMax)
        cycDistMin = min(dist,cycDistMin)
        #seti = [nodeDict[i] for i in pathi0]
        #setj = [nodeDict[i] for i in pathj0]
        #print(seti)
        #print(setj)
        #dist0 = 0
        #for z in range(1, len(nodeDict[samePi])):
        #    for leaf in lvs:
        #        dist0 += abs(int(leaf in seti[0]) * z / len(seti[0]) - int(leaf in setj[0]) * z / len(setj[0]))
        #    if z >= len(seti[0]):
        #        seti.pop(0)
        #    if z >= len(setj[0]):
        #        setj.pop(0)
        #seti = [nodeDict[i] for i in pathi1]
        #setj = [nodeDict[i] for i in pathj1]
        #dist1 = 0
        #for z in range(1, len(nodeDict[samePi])):
        #    for leaf in lvs:
        #        dist1 += abs(int(leaf in seti[0]) * z / len(seti[0]) - int(leaf in setj[0]) * z / len(setj[0]))
        #    if z >= len(seti[0]):
        #        seti.pop(0)
        #    if z >= len(setj[0]):
        #        setj.pop(0)

    #print([max_nr_pick_before_leaf, min_nr_pick_before_leaf, max_length_same_par_l1, min_length_same_par_l1, max_length_same_par_l2, min_length_same_par_l2],pair)
    return [max_nr_pick_before_leaf, min_nr_pick_before_leaf, max_length_same_par_l1, min_length_same_par_l1, max_length_same_par_l2, min_length_same_par_l2, isCherry,blocked,distPPmin,distPPmax,cycDistMin,cycDistMax]

def argmin(lst):
  return lst.index(min(lst))

def getLabelTreeSet(treeSet):
    """
    :param treeSet:
    :return:
    """

    leaves = getLeaves(treeSet[0])
    root = getRoot(treeSet[0])
    nodeDict = dict()
    nodeDict[root] = allChildren(root,treeSet[0])
    for leaf in leaves:
        nodeDict[leaf] = allChildren(leaf,treeSet[0])

    for tree in treeSet:
        for node in treeSet[tree]:
            check = allChildren(node,treeSet[tree])
            if check in nodeDict.values():
                if node not in nodeDict:
                    return None
                if nodeDict[node] != check:
                    return None
            else:
                nodeDict[node] = check

    return nodeDict.copy()

def getPred(treeSet,leaf):
    predList = dict()
    for i in treeSet:
        toDo = [leaf]
        predListB = [leaf]
        while len(toDo) > 0:
            item = toDo.pop(0)
            for pred in treeSet[i].predecessors(item):
                if pred not in predListB:
                    predListB.append(pred)
                    toDo.append(pred)
        predList[i] = predListB
    return predList