import torch
from torch.nn import Linear, Conv1d
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, GATConv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from DataSetTemp import *

dataset = TUDataset(root='data/TUDataset', name='MUTAG')



data = dataset[0]
print(data)
print(data.edge_index)

print(data.x)

print(data.edge_attr)