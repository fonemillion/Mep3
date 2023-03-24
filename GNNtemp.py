import torch
from torch.nn import Linear, Conv1d
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, GATConv, TransformerConv
from torch_geometric.nn import global_mean_pool, global_max_pool
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from DataSetTemp import *

#dataset = TUDataset(root='data/TUDataset', name='MUTAG')

#dataset = MyUltDataset('ult')

dataset = MyOwnDataset('temp') # temp temp2 temp3 temp4
#dataset = MyOwnDataset('temp2')
#dataset = MyOwnDataset('temp3')

torch.manual_seed(12345)
#dataset = dataset.shuffle()

nrTemp = 0
nrNonTemp = 0
for i in range(900):
    if int(dataset[i].y) == 0:
        nrNonTemp += 1
    else:
        nrTemp += 1
print(nrNonTemp,nrTemp)




train_dataset = dataset[:900]
test_dataset = dataset[900:]

#train_dataset = dataset[:1000]
#test_dataset = dataset[1000:]

train_loader = DataLoader(train_dataset, batch_size=900, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        #torch.manual_seed(12345)
        #GATConv
        #SAGEConv
        #TransformerConv
        #NNConv
        self.conv1 = TransformerConv(dataset.num_node_features, hidden_channels)
        self.conv2 = TransformerConv(hidden_channels, hidden_channels)
        self.conv3 = TransformerConv(hidden_channels, hidden_channels)
        self.conv4 = TransformerConv(hidden_channels, hidden_channels)
        self.conv5 = TransformerConv(hidden_channels, hidden_channels)
        #self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        #self.conv2 = GCNConv(hidden_channels, hidden_channels)
        #self.conv2 = GraphConv(hidden_channels, hidden_channels)
        #self.conv3 = SAGEConv((hidden_channels, hidden_channels), hidden_channels)
        #self.conv4 = GCNConv(hidden_channels, hidden_channels)
        #self.conv3 = SAGEConv(hidden_channels, hidden_channels,aggr = 'max')
        #self.conv4 = GCNConv(hidden_channels, hidden_channels)
        #self.lin = Linear(hidden_channels, dataset.num_classes)
        #self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        #self.conv2 = GCNConv(hidden_channels, hidden_channels)
        #self.conv2 = GraphConv(hidden_channels, hidden_channels)
        #self.conv3 = SAGEConv((hidden_channels, hidden_channels), hidden_channels)
        #self.conv4 = GCNConv(hidden_channels, hidden_channels)
        #self.conv3 = SAGEConv(hidden_channels, hidden_channels,aggr = 'max')
        #self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)


    def forward(self, x, edge_index, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        x = x.relu()
        x = self.conv5(x, edge_index)
        #x = x.relu()
        #x = self.conv4(x, edge_index)

        # 2. Readout layer
        #x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        x = global_max_pool(x, batch)
        # 3. Apply a final classifier
        #x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)
        return x


model = GCN(hidden_channels=10)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss(weight= torch.tensor([nrTemp, nrNonTemp], dtype=torch.float))

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.batch)  # Perform a single forward pass.
         loss = criterion(out, data.y)  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.

def test(loader):
     model.eval()

     TotTemp = 0
     correctTemp = 0
     TotNTemp = 0
     correctNTemp = 0
     for data in loader:  # Iterate in batches over the training/test dataset.
         out = model(data.x, data.edge_index, data.batch)
         pred = out.argmax(dim=1)  # Use the class with highest probability.
         for i in range(len(data.y)):
             if int(data.y[i]) == 0:
                 TotNTemp += 1
                 correctNTemp += int((pred[i] == data.y[i]).sum())
             else:
                 TotTemp += 1
                 correctTemp += int((pred[i] == data.y[i]).sum())
         #correct += int((pred == data.y).sum())  # Check against ground-truth labels.
     #return correct / len(loader.dataset)  # Derive ratio of correct predictions.
     return correctNTemp / TotNTemp, correctTemp / TotTemp


x = []
trainAccTemp = []
trainAccNTemp = []
testAccTemp = []
testAccNTemp = []
for epoch in range(1, 201):
    train()
    #if epoch % 10 == 0 :
    if True :
        train_accNT, train_accT = test(train_loader)
        test_accNT, test_accT = test(test_loader)
        x.append(epoch)
        trainAccNTemp.append(train_accNT)
        trainAccTemp.append(train_accT)
        testAccNTemp.append(test_accNT)
        testAccTemp.append(test_accT)
    #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    print(f'Epoch: {epoch:03d}')



import matplotlib.pyplot as plt
plt.plot(x,trainAccNTemp , linestyle = 'dotted', color = 'b' )
plt.plot(x,trainAccTemp , linestyle = 'dotted' , color = 'r')
plt.plot(x,testAccNTemp, color = 'b' )
plt.plot(x,testAccTemp, color = 'r' )
plt.legend(['train_NT', 'train_T', 'test_NT', 'test_T'])
plt.show()