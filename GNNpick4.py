import torch
from torch.nn import Linear, Conv1d
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, GATConv, MessagePassing, TransformerConv, NNConv, \
    GATv2Conv, InstanceNorm, GraphNorm
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from DataSetGen import *
import matplotlib.pyplot as plt
from torch_geometric.nn import MessagePassing, GeneralConv, PDNConv, PNAConv
import numpy as np
from layers import ownLayer, ownLayer2



class clique(torch.nn.Module):
    def __init__(self, xf, ef, out):
        super(clique, self).__init__()
        # GATConv
        # TransformerConv
        # GATv2Conv
        # "add", "sum" "mean", "min", "max" or "mul"
        self.l0 = Linear(xf,xf)
        self.l0_2 = Linear(xf,10)
        self.l0e = Linear(ef,ef)
        self.n = GraphNorm(xf)
        self.n2 = GraphNorm(ef)
        self.l1 = ownLayer(10+xf, 10, ef)
        self.l2 = ownLayer(10+xf, 10, ef)
        self.l3 = ownLayer(10+xf, 10, ef)
        self.l4 = ownLayer(10+xf, 10, ef)
        self.l5 = torch.nn.Linear(10,out)


    def forward(self, x, z, edge_index, z1edge_index, z2edge_index, z3edge_index,  edge_attr, pickable):
        x1 = x
        x1 = self.l0(x1)
        x1 = self.n(x1)
        x1 = self.l0_2(x1)
        x1 = x1.relu()

        edge_attr = self.l0e(edge_attr)
        edge_attr = self.n2(edge_attr)

        x1 = torch.hstack((x, x1))
        x1 = self.l1(x1, edge_index, edge_attr)
        x1 = x1.relu()
        x1 = torch.hstack((x,x1))
        x1 = self.l2(x1, edge_index, edge_attr)
        x1 = x1.relu()
        x1 = torch.hstack((x,x1))
        x1 = self.l3(x1, edge_index, edge_attr)
        x1 = x1.relu()
        x1 = torch.hstack((x,x1))
        x1 = self.l4(x1, edge_index, edge_attr)
        x1 = x1.relu()
        #x1.relu()
        x1 = self.l5(x1)

        x1 = x1[pickable]
        x1 = F.softmax(x1,dim=1)
        #x = F.softmin(x,dim=1)


        return x1

class netw(torch.nn.Module):
    def __init__(self, zf, out):
        super(netw, self).__init__()
        # GATConv
        # TransformerConv
        # GATv2Conv
        self.node_encz1 = Linear(zf, 10)
        self.convz1 = ownLayer2(10, 10,aggr = 'add')
        self.convz2 = ownLayer2(10, 10,aggr = 'add')

        self.linxz = Linear(10, 10)
        self.convxz1 = ownLayer2(10, 10, aggr = 'mean')
        self.convxz2 = ownLayer2(10, 10, aggr = 'mean')
        self.convxz3 = ownLayer2(10, 10, aggr = 'mean')
        self.convxz4 = ownLayer2(10, 10, aggr = 'mean')
        self.convxz4 = ownLayer2(10, 10, aggr = 'mean')
        self.convxz5 = ownLayer2(10, 10, aggr = 'mean')
        #self.linL = Linear(10, 10)
        #self.convleaf = PDNConv(10, 10, edge_dim = 10, hidden_channels = 10)
        #self.conv1 = GATCGATv2Convonv(-1, hidden_channels, edge_dim = dataset.num_edge_features)
        self.lin = Linear(10, out)

    def forward(self, x, z, edge_index, z1edge_index, z2edge_index, z3edge_index,  edge_attr, pickable):
        # 1. Obtain node embeddings
        #y = self.edge_encoder(edge_attr)
        #x = x.relu()
        x = self.node_encz1(z)
        #z = z.relu()



        x = self.convz1(x, z1edge_index)
        x = x.relu()
        x = self.convz2(x, z1edge_index)


        #x = self.linxz(x)
        x = self.convxz1(x,z1edge_index)
        x = x.relu()
        #x = self.convxz2(x,z2edge_index)
        #x = x.relu()
        #x = self.convxz3(x,z1edge_index)
        #x = x.relu()
        #x = self.convxz4(x,z3edge_index)
        #x = x.relu()
        #x = self.convxz5(x,z1edge_index)
        #x = x.relu()

        #x = self.convleaf(x, edge_index, edge_attr = edge_attr)

        x = self.lin(x)
        x = x[pickable]
        x = F.softmax(x,dim=1)
        #x = F.softmin(x,dim=1)

        # Normalize the features by dividing each element by the total


        return x

class comb(torch.nn.Module):
    def __init__(self, zf, xf, ef, out):
        super(comb, self).__init__()
        # GATConv
        # TransformerConv
        # GATv2Conv



        self.node_encz1 = Linear(zf, 10)
        self.convz1 = TransformerConv(10, 10)
        self.convz2 = TransformerConv(10, 10)

        self.linxz = Linear(10, 10)
        self.convxz1 = GeneralConv(10, 10)
        self.convxz2 = GATv2Conv(10, 10)
        self.convxz3 = GATv2Conv(10, 10)
        #self.convxz3A = GATv2Conv(10, 10)
        #self.convxz3B = GATv2Conv(10, 10)
        self.convxz4 = GATv2Conv(10, 10)
        self.convxz4 = SAGEConv(10, 10, aggr = 'max')
        self.convxz5 = GATv2Conv(10, 10)
        #self.linL = Linear(10, 10)
        #self.convleaf = PDNConv(10, 10, edge_dim = 10, hidden_channels = 10)
        #self.conv1 = GATCGATv2Convonv(-1, hidden_channels, edge_dim = dataset.num_edge_features)
        self.l1 = ownLayer(xf, 10, ef)
        self.l2 = ownLayer(10+xf, 10, ef)
        self.l3 = ownLayer(10, 10, ef)
        self.lin = Linear(20, out)

    def forward(self, x, z, edge_index, z1edge_index, z2edge_index, z3edge_index,  edge_attr, pickable):
        # 1. Obtain node embeddings
        #y = self.edge_encoder(edge_attr)
        #x = x.relu()
        z = self.node_encz1(z)
        #z = z.relu()



        z = self.convz1(z, z1edge_index)
        z = z.relu()
        z = self.convz2(z, z1edge_index)


        z = self.linxz(z)
        z = self.convxz1(z,z1edge_index)
        z = z.relu()
        z = self.convxz2(z,z2edge_index)
        z = z.relu()
        z = self.convxz3(z,z1edge_index)
        z = z.relu()
        z = self.convxz4(z,z3edge_index)
        z = z.relu()
        z = self.convxz5(z,z1edge_index)
        #x = x.relu()

        x1 = self.l1(x, edge_index, edge_attr)
        x1.relu()
        x = torch.hstack((x,x1))
        x = self.l2(x, edge_index, edge_attr)
        x.relu()
        x = self.l3(x, edge_index, edge_attr)
        x.relu()

        x = torch.hstack((x, z))
        #x = self.convleaf(x, edge_index, edge_attr = edge_attr)

        x = self.lin(x)
        x = x[pickable]
        x = F.softmax(x,dim=1)
        #x = F.softmin(x,dim=1)

        # Normalize the features by dividing each element by the total


        return x


if __name__ == '__main__':
    def train():
        model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            out = model(data.x, data.z, data.edge_index, data.z1edge_index, data.z2edge_index, data.z3edge_index, data.edge_attr, data.pickable)  # Perform a single forward pass.
            loss = criterion(out, data.y[data.pickable])  # Compute the loss.
            loss.backward()  # Derive gradients.
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

        return float(loss)
    def test(loader):
        model.eval()
        returnData = []
        for i in range(dataset.num_classes):
            returnData.append([0, 0])

        # torch.save(model, PATH)
        for data in loader:  # Iterate in batches over the training/test dataset.
            out = model(data.x, data.z, data.edge_index, data.z1edge_index, data.z2edge_index, data.z3edge_index, data.edge_attr, data.pickable)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            #print(pred)
            y = data.y[data.pickable]
            #print(y)
            for i in range(len(y)):
                returnData[int(y[i])][0] += 1
                returnData[int(y[i])][1] += int((pred[i] == y[i]).sum())
        ratio = []
        for i in range(dataset.num_classes):
            if returnData[i][0] == 0:
                ratio.append(-1)
            else:
                ratio.append(returnData[i][1] / returnData[i][0])

        return ratio

    dataset = MyOwnDataset('DataGen/ret_L15_T2', sample = 200, delete = False)
    #dataset = MyOwnDataset('DataGen/ret_L15_T3', sample = 200, delete = True)
    #dataset = MyOwnDataset('DataGen/ret_lowret', sample = 200, delete = False)
    #dataset = MyOwnDataset('DataGen/ret_cyc', sample = 200, delete = False)
    #dataset = MyOwnDataset('DataGen/ret_lowmid', sample = 200, delete = False)

    x_f = dataset[0].x.size()[1]
    z_f = dataset[0].z.size()[1]
    ex_f = dataset[0].edge_attr.size()[1]
    num_classes = dataset.num_classes

    #model = GCN()
    model = clique(x_f,ex_f,num_classes)
    #model = netw(z_f,num_classes)
    #model = comb(z_f,x_f,ex_f,num_classes)

    #torch.manual_seed(12345)
    dataset = dataset.shuffle()
    #print(dataset)
    print(dataset)
    #print(dataset[0].nodeDict[0])
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

    #train_dataset = dataset[:4000]
    train_dataset = dataset



    train_loader = DataLoader(train_dataset, batch_size=1000, shuffle=True)


    # Rprop ASGD Adam
    optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
    criterion = torch.nn.CrossEntropyLoss(weight= torch.tensor([weight[1], weight[0]], dtype=torch.float))
    #criterion = torch.nn.CrossEntropyLoss(weight= torch.tensor([weight[1],weight[0], 0], dtype=torch.float))
    #criterion = torch.nn.CrossEntropyLoss(weight= torch.tensor([1, 1, 0, 1], dtype=torch.float))
    #criterion = torch.nn.CrossEntropyLoss()


    x = []
    ratio_train = []
    #ratio_test = []
    loss_data = []
    bestloss = 100

    i = 0
    while True:
        filename = 'models/ret/model_' + str(i)
        if not os.path.isfile(filename + '.pt'):
            break
        i += 1


    for epoch in range(1, 400):
        losstrain = train()
        #if epoch % 10 == 0 :
        if True :
            loss_data.append(losstrain)
            ratio_train.append(test(train_loader))
            #ratio_test.append(test(test_loader))
            x.append(epoch)
        #print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
        print(f'Epoch: {epoch:03d}')
        if losstrain < bestloss:
            bestloss = losstrain
            #model_scripted = torch.jit.script(model)  # Export to TorchScript
            #model_scripted.save('DataGen/retdata.pt')
            torch.save(model.state_dict(), filename + '.pt')


    model.eval()
    data = dataset[0]
    out = model(data.x, data.z, data.edge_index, data.z1edge_index, data.z2edge_index, data.z3edge_index,
                data.edge_attr, data.pickable)
    print(out)
    print(out.argmax(dim=1))
    print(data.y[data.pickable])

    colours = {0:'b',1:'r',2:'g',3:'y'}
    plt.figure(0)
    for i in range(dataset.num_classes):
        plotSet = [n[i] for n in ratio_train]
        plt.plot(x, plotSet, linestyle ='dotted', color = colours[i])
        #plotSet = [n[i] for n in ratio_test]
        #plt.plot(x, plotSet, color = colours[i])
    #plt.show()

    print(max(np.sum(ratio_train, axis=1)))
    print(min(loss_data))
    plt.figure(1)
    plt.plot(x, loss_data)
    plt.show()
