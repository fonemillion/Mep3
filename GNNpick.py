import torch
from torch.nn import Linear, Conv1d
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, GATConv, MessagePassing, TransformerConv,NNConv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from DataSetPick import *
import matplotlib.pyplot as plt

dataset = MyOwnDataset('DataGen/ret', sample = 2)

#torch.manual_seed(12345)
dataset = dataset.shuffle()
weight = [0,0]
for i in dataset:
    for j in range(len(i.y)):
        if i.x[j][6] == 1:
            weight[int(i.y[j])] += 1
print(weight)

train_dataset = dataset[:90]
test_dataset = dataset[90:]

#train_dataset = dataset[:1000]
#test_dataset = dataset[1000:]

train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        #torch.manual_seed(12345)
        #GATConv
        #SAGEConv
        #TransformerConv
        #NNConv
        self.conv1 = TransformerConv(dataset.num_node_features, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = SAGEConv(hidden_channels, hidden_channels)
        self.conv4 = TransformerConv(hidden_channels, hidden_channels)
        self.conv5 = TransformerConv(hidden_channels, hidden_channels)
        #self.conv1 = GCNConv(dataset.num_node_features, hidden_channels)
        #self.conv2 = GCNConv(hidden_channels, hidden_channels)
        #self.conv2 = GraphConv(hidden_channels, hidden_channels)
        #self.conv3 = SAGEConv((hidden_channels, hidden_channels), hidden_channels)
        #self.conv4 = GCNConv(hidden_channels, hidden_channels)
        #self.conv3 = SAGEConv(hidden_channels, hidden_channels,aggr = 'max')
        #self.conv4 = GCNConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, dataset.num_classes)
        #self.conv5 = GCNConv(hidden_channels, dataset.num_classes)


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
        x = x.relu()

        # 2. Readout layer
        #x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        #x = F.dropout(x, p=0.5, training=self.training)
        #x = self.conv5(x, edge_index)
        x = self.lin(x)
        return x


model = GCN(hidden_channels=10)


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#criterion = torch.nn.CrossEntropyLoss(weight= torch.tensor([weight[1],weight[0]], dtype=torch.float))
criterion = torch.nn.CrossEntropyLoss(weight= torch.tensor([weight[1],weight[0], 0], dtype=torch.float))
#criterion = torch.nn.CrossEntropyLoss(weight= torch.tensor([1, 1, 0, 1], dtype=torch.float))
#criterion = torch.nn.CrossEntropyLoss()

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
    returnData = []
    for i in range(dataset.num_classes):
        returnData.append([0,0])


    #torch.save(model, PATH)
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        #print(pred)
        #print(data.y)
        for i in range(len(data.y)):

            if data.x[i][6] == 1:
                returnData[int(data.y[i])][0] += 1
                returnData[int(data.y[i])][1] += int((pred[i] == data.y[i]).sum())
    ratio = []
    for i in range(dataset.num_classes):
        if returnData[i][0] == 0:
            ratio.append(0)
        else:
            ratio.append(returnData[i][1] / returnData[i][0])

    return ratio

x = []
ratio_train = []
ratio_test = []
for epoch in range(1, 200):
    train()
    #if epoch % 10 == 0 :
    if True :
        ratio_train.append(test(train_loader))
        ratio_test.append(test(test_loader))
        x.append(epoch)
    #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    print(f'Epoch: {epoch:03d}')

colours = {0:'b',1:'r',2:'g',3:'y'}

for i in range(dataset.num_classes):
    plotSet = [n[i] for n in ratio_train]
    plt.plot(x, plotSet, linestyle ='dotted', color = colours[i])
    plotSet = [n[i] for n in ratio_test]
    plt.plot(x, plotSet, color = colours[i])
plt.show()