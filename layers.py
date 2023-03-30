import math

import torch
from torch.nn import Linear, Parameter
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

from torch_geometric.utils import softmax

class ownLayer(MessagePassing):
    def __init__(self, in_channels, out_channels, edge_channels, aggr = 'add'):
        super().__init__(aggr)
        hidden_channels = 10


        self.feature_module_final = torch.nn.Sequential(
            torch.nn.Linear(2*in_channels + edge_channels, hidden_channels),
            torch.nn.Sigmoid()
        )
        self.feature_module_final2 = torch.nn.Sequential(
            torch.nn.Linear(2*in_channels + edge_channels, hidden_channels),
            torch.nn.Sigmoid()
        )

        self.alpha = torch.nn.Sequential(
            #PreNormLayer(1, shift=False),
            #torch.nn.ReLU(),
            torch.nn.Linear(2*in_channels + edge_channels, hidden_channels)
            #torch.nn.Tanh(),
            #torch.nn.Linear(3*hidden_channels, 1)
            #torch.nn.Tanh(),
            #torch.nn.Linear(hidden_channels, hidden_channels),
            #torch.nn.ReLU()
        )
        self.beta = torch.nn.Sequential(
            torch.nn.Linear(2*in_channels + edge_channels, hidden_channels)
        )
        #self.post_conv_module = torch.nn.Sequential(
        #    PreNormLayer(1, shift=False)
        #)
        self.lin = torch.nn.Linear(in_channels,hidden_channels)
        self.lin2 = torch.nn.Linear(in_channels,hidden_channels)
        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, node_features, edge_indices, edge_features):
        output = self.propagate(edge_indices, node_features = node_features, edge_features=edge_features)
        output = output + self.lin(node_features).sigmoid()*self.lin2(node_features).sigmoid()

        #return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))
        return self.output_module(output)


    #def message(self, x_j, norm):
    #    # x_j has shape [E, out_channels]

    #    # Step 4: Normalize node features.
    #    return norm.view(-1, 1) * x_j


    def message(self, node_features_i, node_features_j, edge_features, ptr, index):
        #print(node_features_i.shape,node_features_j.shape,edge_features.shape)
        output = self.feature_module_final(torch.hstack((node_features_i, edge_features , node_features_j)))
        output2 = self.feature_module_final2(torch.hstack((node_features_i, edge_features , node_features_j)))
        output = output * output2

        alpha = self.alpha(torch.hstack((node_features_i, edge_features , node_features_j)))
        beta = self.beta(torch.hstack((node_features_i, edge_features , node_features_j)))

        alpha = alpha*beta

        alpha = softmax(alpha, index, ptr)
        #print(alpha.size())
        output = output * alpha
        #print(output.size())
        return output

class ownLayer2(MessagePassing):
    def __init__(self, in_channels, out_channels, aggr = 'max'):
        super().__init__(aggr)
        hidden_channels = 10

        self.feature_module_left = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels)
        )
        self.feature_module_right = torch.nn.Sequential(
            torch.nn.Linear(in_channels, hidden_channels)
        )
        self.feature_module_final = torch.nn.Sequential(
            #PreNormLayer(1, shift=False),
            #torch.nn.ReLU(),
            #torch.nn.Linear(2*hidden_channels, 2*hidden_channels),
            #torch.nn.Tanh(),
            torch.nn.Linear(2*hidden_channels, hidden_channels),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden_channels, hidden_channels),
            torch.nn.ReLU()
        )

        #self.post_conv_module = torch.nn.Sequential(
        #    PreNormLayer(1, shift=False)
        #)

        # output_layers
        self.output_module = torch.nn.Sequential(
            torch.nn.Linear(hidden_channels, out_channels),
        )

    def forward(self, node_features, edge_indices):
        output = self.propagate(edge_indices, node_features = node_features)
        #return self.output_module(torch.cat([self.post_conv_module(output), right_features], dim=-1))
        return self.output_module(output)

    #def message(self, x_j, norm):
    #    # x_j has shape [E, out_channels]

    #    # Step 4: Normalize node features.
    #    return norm.view(-1, 1) * x_j


    def message(self, node_features_i, node_features_j):
        #print(node_features_i.shape,node_features_j.shape,edge_features.shape)
        xi = self.feature_module_left(node_features_i)
        xj = self.feature_module_right(node_features_j)
        #print(xi,xj)
        output = self.feature_module_final(torch.hstack((xi , xj)))
        return output



