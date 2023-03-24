import pickle
from torch_geometric.data import Data
import torch
from torch_geometric.data import InMemoryDataset, download_url
from InstanceGenerators.netGen import *

class MyOwnDataset(InMemoryDataset):
    def __init__(self, name, transform=None, pre_transform=None, pre_filter=None):
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


        for i in range(1000):
            filename = self.name + '/inst_' + str(i) + '.pickle'
            with open(filename, 'rb') as f:
                [treeSet, check] = pickle.load(f)

            net = treeToTN(treeSet)
            list1 = []
            if check:
                y = 1
            else:
                y = 0

            H = net.to_undirected()
            leaves = [u for u in net.nodes() if net.out_degree(u) == 0]
            origin = 2 * len(leaves) - 2
            cycles = nx.cycle_basis(H,origin)
            #del H

            bridgeNodes = []

            for bridge in nx.bridges(H, root=None):
                if bridge[0] not in bridgeNodes:
                    bridgeNodes.append(bridge[0])
                if bridge[1] not in bridgeNodes:
                    bridgeNodes.append(bridge[1] )

            for node in range(net.number_of_nodes()):
                nr = 0
                type = 0
                for cyc in cycles:
                    if node in cyc:
                        nr += 1
                if node in leaves:
                    type = -1
                if node == origin:
                    type = 1
                inBridge = 0
                if node in bridgeNodes:
                    inBridge = 1
                #list1.append([net.out_degree(node), net.in_degree(node),net.out_degree(node) + net.in_degree(node),len(allChildren(node,net)),net.number_of_nodes(),nr])
                list1.append([net.out_degree(node), net.in_degree(node),len(allChildren(node,net)),nr,type,inBridge])


            # add edges to all leaves
            leafpair = [(l1,l2) for l1 in leaves for l2 in leaves if l1 != l2]
            for pair in leafpair:
                H.add_edge(pair[0],pair[1])
            # Convert the graph to a TorchGeometric Data object

            data = Data()
            data.x = torch.tensor(list1, dtype=torch.float)
            edges = list(H.edges) + [(i[1], i[0]) for i in H.edges]
            data.edge_index = torch.tensor(edges).t().contiguous()  # Edge index
            data.y = torch.tensor(y, dtype=torch.long)
            data_list.append(data)
            if data.is_directed():
                raise Exception("graph is directed")
            del H, net

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



class MyUltDataset(InMemoryDataset):
    def __init__(self, name, transform=None, pre_transform=None, pre_filter=None):
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
        for i in range(1000):
            filename = 'temp2/inst_' + str(i) + '.pickle'
            with open(filename, 'rb') as f:
                [treeSet, check] = pickle.load(f)

            net = treeToTN(treeSet)
            list1 = []
            if check:
                y = 1
            else:
                y = 0

            H = net.to_undirected()
            leaves = [u for u in net.nodes() if net.out_degree(u) == 0]
            origin = 2 * len(leaves) - 2
            cycles = nx.cycle_basis(H,origin)
            #del H

            bridgeNodes = []

            for bridge in nx.bridges(H, root=None):
                if bridge[0] not in bridgeNodes:
                    bridgeNodes.append(bridge[0])
                if bridge[1] not in bridgeNodes:
                    bridgeNodes.append(bridge[1] )

            for node in range(net.number_of_nodes()):
                nr = 0
                type = 0
                for cyc in cycles:
                    if node in cyc:
                        nr += 1
                if node in leaves:
                    type = -1
                if node == origin:
                    type = 1
                inBridge = 0
                if node in bridgeNodes:
                    inBridge = 1
                #list1.append([net.out_degree(node), net.in_degree(node),net.out_degree(node) + net.in_degree(node),len(allChildren(node,net)),net.number_of_nodes(),nr])
                list1.append([net.out_degree(node), net.in_degree(node),len(allChildren(node,net)),nr,type,inBridge])


            # add edges to all leaves
            leafpair = [(l1,l2) for l1 in leaves for l2 in leaves if l1 != l2]
            for pair in leafpair:
                net.add_edge(pair[0],pair[1])
            # Convert the graph to a TorchGeometric Data object
            data = Data()
            data.x = torch.tensor(list1, dtype=torch.float)
            data.edge_index = torch.tensor(list(H.edges)).t().contiguous()  # Edge index
            data.y = torch.tensor(y, dtype=torch.long)
            data_list.append(data)
        for i in range(1000):
            filename = 'temp/inst_' + str(i) + '.pickle'
            with open(filename, 'rb') as f:
                [treeSet, check] = pickle.load(f)

            net = treeToTN(treeSet)
            list1 = []
            if check:
                y = 1
            else:
                y = 0

            H = net.to_undirected()
            leaves = [u for u in net.nodes() if net.out_degree(u) == 0]
            origin = 2 * len(leaves) - 2
            cycles = nx.cycle_basis(H,origin)
            #del H

            bridgeNodes = []

            for bridge in nx.bridges(H, root=None):
                if bridge[0] not in bridgeNodes:
                    bridgeNodes.append(bridge[0])
                if bridge[1] not in bridgeNodes:
                    bridgeNodes.append(bridge[1] )

            for node in range(net.number_of_nodes()):
                nr = 0
                type = 0
                for cyc in cycles:
                    if node in cyc:
                        nr += 1
                if node in leaves:
                    type = -1
                if node == origin:
                    type = 1
                inBridge = 0
                if node in bridgeNodes:
                    inBridge = 1
                #list1.append([net.out_degree(node), net.in_degree(node),net.out_degree(node) + net.in_degree(node),len(allChildren(node,net)),net.number_of_nodes(),nr])
                list1.append([net.out_degree(node), net.in_degree(node),len(allChildren(node,net)),nr,type,inBridge])


            # add edges to all leaves
            leafpair = [(l1,l2) for l1 in leaves for l2 in leaves if l1 != l2]
            for pair in leafpair:
                net.add_edge(pair[0],pair[1])
            # Convert the graph to a TorchGeometric Data object
            data = Data()
            data.x = torch.tensor(list1, dtype=torch.float)
            data.edge_index = torch.tensor(list(H.edges)).t().contiguous()  # Edge index
            data.y = torch.tensor(y, dtype=torch.long)
            data_list.append(data)

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])



