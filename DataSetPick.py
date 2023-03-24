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
    def __init__(self, name, transform=None, pre_transform=None, pre_filter=None, sample = False):
        dir = name + 'data'

        if os.path.exists(dir):
            shutil.rmtree(dir)
            print(f"{dir} deleted successfully.")
        else:
            print(f"{dir} does not exist.")

        self.sample = sample
        self.name = name
        super().__init__(name + 'data', transform, pre_transform, pre_filter)
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


        if self.sample == 'all':
            i = 0
            while True:
                filename = self.name + '/inst_' + str(i) + '.pickle'
                if not os.path.isfile(filename):
                    break
                data = toData2(filename)
                if data != None:
                    data_list.append(data)
                i += 1
        else:
            if self.sample == False:
                N = 1000
            else:
                N = self.sample

            for i in range(N):
                filename = self.name + '/inst_' + str(i) + '.pickle'
                if not os.path.isfile(filename):
                    raise Exception("file not found")
                data = toData2(filename)
                data_list.append(data)




        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


def toData(filename):

    with open(filename, 'rb') as f:
        [treeSet, retValues] = pickle.load(f)

    nodeDict = reLabelTreeSet(treeSet)
    for leaf in retValues:
        leaf[0] = nodeDict.index([leaf[0]])

    nonTemp = []
    for leaf in retValues:
        if leaf[1] == 0:
            nonTemp.append(leaf[0])
    retValues = [i for i in retValues if i[0] not in nonTemp]


    for leaf in retValues:
        leaf[1] += weightLeaf(treeSet, leaf[0])
    del nodeDict

    bestW = min([i[1] for i in retValues])
    bestPick = [i[0] for i in retValues if i[1] == bestW]


    net = nx.DiGraph()
    for i in treeSet:
        net.add_nodes_from(treeSet[i].nodes())
        net.add_edges_from(treeSet[i].edges())




    #print(y)
    H = net.to_undirected()
    leaves = [u for u in net.nodes() if net.out_degree(u) == 0]
    origin = 2 * len(leaves) - 2
    cycles = nx.cycle_basis(H, origin)
    # del H

    bridgeNodes = []

    for bridge in nx.bridges(H, root=None):
        if bridge[0] not in bridgeNodes:
            bridgeNodes.append(bridge[0])
        if bridge[1] not in bridgeNodes:
            bridgeNodes.append(bridge[1])



    pickAble = canPick(treeSet)
    list1 = []
    y = []
    for node in range(net.number_of_nodes()):
        nr = 0
        type = 0
        for cyc in cycles:
            if node in cyc:
                nr += 1
        if node in leaves:
            type = 1
        if node == origin:
            type = 0
        inBridge = 0
        if node in bridgeNodes:
            inBridge = 1

        if node in pickAble:
            pickble = 1
        else:
            pickble = 0
        #list1.append([net.out_degree(node), net.in_degree(node),net.out_degree(node) + net.in_degree(node),len(allChildren(node,net)),net.number_of_nodes(),nr,pickble])
        list1.append([net.out_degree(node), net.in_degree(node), len(allChildren(node, net)), nr, type, inBridge,pickble])
        #list1.append([type])
        if node in bestPick:
        #if node in leaves:
            y.append(0)
        #elif node in nonTemp:
        #    y.append(3)
        elif node in pickAble:
            y.append(1)
        else:
            y.append(2)
        #if node in bestPick:
        #    y.append(1)
        #else:
        #    y.append(0)
    # add edges to all leaves
    leafpair = [(l1, l2) for l1 in leaves for l2 in leaves if l1 != l2]
    for pair in leafpair:
        H.add_edge(pair[0], pair[1])
    # Convert the graph to a TorchGeometric Data object

    data = Data()
    data.x = torch.tensor(list1, dtype=torch.float)

    edges = list(H.edges) + [(i[1],i[0]) for i in H.edges]
    data.edge_index = torch.tensor(edges).t().contiguous()  # Edge index
    data.y = torch.tensor(y, dtype=torch.long)
    if data.is_directed():
        raise Exception("graph is directed")
    del H,net
    return data




def toData2(filename, type = 'file'):

    if type == 'file':
        with open(filename, 'rb') as f:
            [treeSet, retValues] = pickle.load(f)
    else:
        [treeSet, retValues] = filename
    # getting y

    nodeDict = reLabelTreeSet(treeSet)

    if retValues != None:
        # remove wrong data
        if min([i[1] for i in retValues if i[1] != 0]) == 1:
            return None

        for leaf in retValues:
            leaf[0] = nodeDict.index([leaf[0]])

        nonTemp = []
        for leaf in retValues:
            if leaf[1] == 0:
                nonTemp.append(leaf[0])
        retValues = [i for i in retValues if i[0] not in nonTemp]


        for leaf in retValues:
            leaf[1] += weightLeaf(treeSet, leaf[0])
        #del nodeDict

        bestW = min([i[1] for i in retValues])
        bestPick = [i[0] for i in retValues if i[1] == bestW]
    else:
        bestPick = []


    net = nx.DiGraph()
    for i in treeSet:
        net.add_nodes_from(treeSet[i].nodes())
        net.add_edges_from(treeSet[i].edges())


    x = []
    pickset = []
    y = []
    root = getRoot(net)

    lvs = getLeaves(net)
    lvs = range(1,len(lvs)+1)
    #print(lvs)
    pickAble = canPick(treeSet)
    for leaf in lvs:
        if leaf in pickAble:
            pickble = 1
        else:
            pickble = 0
        min_to_root = len(lvs)+2
        max_to_root = 0
        for tree in treeSet:
            length = findLengthRoot(treeSet[tree], leaf)
            min_to_root = min(min_to_root,length)
            max_to_root = max(max_to_root,length)
        x.append([min_to_root,max_to_root,pickble])
        pickset.append(leaf in pickAble)

        if leaf in bestPick:
            y.append(0)
        #elif leaf in pickAble:
        #    y.append(1)
        else:
            y.append(1)
            #y.append(2)


    edge_attr = []
    edge_index = []
    leafpairs = [(n,m) for n in lvs for m in lvs if n != m]
    for pair in leafpairs:
        #leaf1 = pair[0]-1
        #leaf2 = pair[1]-1
        edge_attr.append(getEdgeFeat(nodeDict, treeSet, pair))
        edge_index.append([pair[0]-1,pair[1]-1]) # -1 because no root


    # Convert the graph to a TorchGeometric Data object
    data = Data()

    data.x = torch.tensor(x, dtype=torch.float)
    data.edge_index = torch.tensor(edge_index).t().contiguous()
    data.edge_attr = torch.tensor(edge_attr, dtype=torch.float)
    data.y = torch.tensor(y, dtype=torch.long)
    data.pickable = torch.tensor(pickset, dtype=torch.bool)

    #print(data.edge_index)
    #if data.is_directed():
    #    raise Exception("graph is directed")
    del net, nodeDict
    return data


def findLengthRoot(tree, leaf):
    root = getRoot(tree)
    lvs = getLeaves(tree)
    if leaf not in lvs:
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

def getEdgeFeat(nodeDict,treeSet,pair):
    lvs = getLeaves(treeSet[0])

    max_nr_pick_before_leaf = 0
    min_nr_pick_before_leaf = len(lvs)

    max_length_same_par_l1 = 0
    min_length_same_par_l1 = len(lvs)

    max_length_same_par_l2 = 0
    min_length_same_par_l2 = len(lvs)

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

    return [max_nr_pick_before_leaf, min_nr_pick_before_leaf, max_length_same_par_l1, min_length_same_par_l1, max_length_same_par_l2, min_length_same_par_l2]

def argmin(lst):
  return lst.index(min(lst))

