import os
import pickle
import copy
from torch.nn import Linear
import torch
from torch_geometric.nn import GATv2Conv
from DataGen.pickClass import canPick, weightLeaf, reLabelTreeSet, pickCher, getLeaves
import torch.nn.functional as F
from DataSetPick import MyOwnDataset, toData2
import random
from Solvers2.TreeAn import showTree, pickRipe
from Solvers2.GreedyPicker import greedyPick


def GNNheuristic(treeSet, type = 'max'):

    dataset = MyOwnDataset('DataGen/ret', sample = 1)
    class GCN(torch.nn.Module):
        def __init__(self, hidden_channels):
            super(GCN, self).__init__()
            self.conv1 = GATv2Conv(dataset.num_node_features, hidden_channels, edge_dim = dataset.num_edge_features)
            self.conv2 = GATv2Conv(hidden_channels, hidden_channels, edge_dim = dataset.num_edge_features)
            self.conv3 = GATv2Conv(hidden_channels, hidden_channels, edge_dim = dataset.num_edge_features)
            self.conv4 = GATv2Conv(hidden_channels, hidden_channels, edge_dim = dataset.num_edge_features)
            #self.conv1 = GATConv(-1, hidden_channels, edge_dim = dataset.num_edge_features)
            self.edge_encoder = Linear(dataset.num_edge_features, dataset.num_edge_features)
            self.edge_encoder2 = Linear(dataset.num_edge_features, dataset.num_edge_features)
            self.lin = Linear(hidden_channels, dataset.num_classes)

        def forward(self, x, edge_index, edge_attr, pickable):
            # 1. Obtain node embeddings
            #y = self.edge_encoder(edge_attr)
            edge_attr = self.edge_encoder(edge_attr)
            edge_attr = self.edge_encoder2(edge_attr)
            x = self.conv1(x, edge_index, edge_attr = edge_attr)
            x = x.relu()
            x = self.conv2(x, edge_index, edge_attr = edge_attr)
            x = x.relu()
            x = self.conv3(x, edge_index, edge_attr = edge_attr)
            x = x.relu()
            x = self.conv4(x, edge_index, edge_attr = edge_attr)
            x = self.lin(x)
            x = x[pickable]
            x = F.softmax(x,dim=1)
            #x = F.log_softmax(x,dim=1)

            return x
    model = GCN(hidden_channels=10)
    filename = 'models/ret/model_6'
    model.load_state_dict(torch.load(filename + '.pt'))
    model.eval()
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
            data = toData2([treeSet, None], type='tree')
            pickSet = canPick(treeSet)
            pickSet.sort()
            weight = []
            out = model(data.x, data.edge_index, data.edge_attr, data.pickable)
            for i in range(len(pickSet)):
                weight.append(float(out[i][0]))
            if type == 'max':
                leaf = pickSet[argmax(weight)]
            else:
                leaf = random.choices(pickSet, weights=weight, k=1)[0]
            # print("multiple choise ", leaf, pickSet, weight)
            ret += weightLeaf(treeSet, leaf)
            pickCher(treeSet, leaf)

    return ret




def argmax(lst):
  return lst.index(max(lst))

