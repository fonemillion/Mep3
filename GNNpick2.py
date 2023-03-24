import torch
from torch.nn import Linear, Conv1d
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, GATConv, MessagePassing, TransformerConv,NNConv, GATv2Conv
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.datasets import TUDataset
from DataSetPick import *
import matplotlib.pyplot as plt
from torch_geometric.nn import MessagePassing, GeneralConv, PDNConv, PNAConv

dataset = MyOwnDataset('DataGen/ret', sample = 1000)
#torch.manual_seed(12345)
dataset = dataset.shuffle()
print(dataset)
weight = [0,0]
for i in dataset:
    for j in range(len(i.y)):
        if i.pickable[j] == 1:
            weight[int(i.y[j])] += 1
print(weight)


nonweight = [0,0]
for i in dataset:
    for j in range(len(i.y)):
        if i.pickable[j] == 0:
            nonweight[int(i.y[j])] += 1
print(nonweight)

train_dataset = dataset[:4000]
test_dataset = dataset[4000:]

#train_dataset = dataset[:1000]
#test_dataset = dataset[1000:]

train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)


class GCN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(GCN, self).__init__()
        #self.conv1 = GATv2Conv(dataset.num_node_features, hidden_channels, edge_dim = dataset.num_edge_features)
        #self.conv2 = GATv2Conv(hidden_channels, hidden_channels, edge_dim = dataset.num_edge_features)
        #self.conv3 = GATv2Conv(hidden_channels, hidden_channels, edge_dim = dataset.num_edge_features)
        #self.conv4 = GATv2Conv(hidden_channels, hidden_channels, edge_dim = dataset.num_edge_features)
        self.conv1 = PDNConv(dataset.num_node_features, hidden_channels, edge_dim = dataset.num_edge_features, hidden_channels=hidden_channels)
        self.conv2 = PDNConv(hidden_channels, hidden_channels, edge_dim = dataset.num_edge_features, hidden_channels=hidden_channels)
        self.conv3 = PDNConv(hidden_channels, hidden_channels, edge_dim = dataset.num_edge_features, hidden_channels=hidden_channels)
        self.conv4 = PDNConv(hidden_channels, hidden_channels, edge_dim = dataset.num_edge_features, hidden_channels=hidden_channels)
        #self.convx2 = PDNConv(10, 10, edge_dim=10, hidden_channels=10)
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


optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = torch.nn.CrossEntropyLoss(weight= torch.tensor([weight[1],weight[0]], dtype=torch.float))
#criterion = torch.nn.CrossEntropyLoss(weight= torch.tensor([weight[1],weight[0], 0], dtype=torch.float))
#criterion = torch.nn.CrossEntropyLoss(weight= torch.tensor([1, 1, 0, 1], dtype=torch.float))
#criterion = torch.nn.CrossEntropyLoss()

def train():
    model.train()

    for data in train_loader:  # Iterate in batches over the training dataset.
         out = model(data.x, data.edge_index, data.edge_attr,data.pickable)  # Perform a single forward pass.
         loss = criterion(out, data.y[data.pickable])  # Compute the loss.
         loss.backward()  # Derive gradients.
         optimizer.step()  # Update parameters based on gradients.
         optimizer.zero_grad()  # Clear gradients.


    return float(loss)


def test(loader):
    model.eval()
    returnData = []
    for i in range(dataset.num_classes):
        returnData.append([0,0])


    #torch.save(model, PATH)
    for data in loader:  # Iterate in batches over the training/test dataset.
        out = model(data.x, data.edge_index, data.edge_attr, data.pickable)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        #print(pred)
        #print(data.y)
        y = data.y[data.pickable]
        for i in range(len(y)):

            #if data.pickable[i] == 1:
            returnData[int(y[i])][0] += 1
            returnData[int(y[i])][1] += int((pred[i] == y[i]).sum())
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
loss_data = []
bestloss = 100

i = 0
while True:
    filename = 'models/ret/model_' + str(i)
    if not os.path.isfile(filename + '.pt'):
        break
    i += 1


for epoch in range(1, 100):
    losstrain = train()
    #if epoch % 10 == 0 :
    if True :
        loss_data.append(losstrain)
        ratio_train.append(test(train_loader))
        ratio_test.append(test(test_loader))
        x.append(epoch)
    #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    print(f'Epoch: {epoch:03d}')
    if losstrain < bestloss:
        bestloss = losstrain
        #model_scripted = torch.jit.script(model)  # Export to TorchScript
        #model_scripted.save('DataGen/retdata.pt')
        torch.save(model.state_dict(), filename + '.pt')


colours = {0:'b',1:'r',2:'g',3:'y'}
plt.figure(0)
for i in range(dataset.num_classes):
    plotSet = [n[i] for n in ratio_train]
    plt.plot(x, plotSet, linestyle ='dotted', color = colours[i])
    plotSet = [n[i] for n in ratio_test]
    plt.plot(x, plotSet, color = colours[i])

plt.figure(1)
plt.plot(x, loss_data)
plt.show()