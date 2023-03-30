import torch
from torch.nn import Linear, Conv1d
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, SAGEConv, GraphConv, GATConv, MessagePassing, TransformerConv, NNConv, \
    GATv2Conv, GraphNorm
from torch_geometric.nn import global_mean_pool
from torch_geometric.loader import DataLoader
from DataSetGen import *
import matplotlib.pyplot as plt
from torch_geometric.nn import MessagePassing, GeneralConv, PDNConv, PNAConv
import numpy as np

class GCN(torch.nn.Module):
    def __init__(self):
        super(GCN, self).__init__()
        # GATConv
        # TransformerConv
        # GATv2Conv
        self.node_encx1 = Linear(3, 10)
        self.node_encz1 = Linear(4, 10)
        self.convz1 = TransformerConv(10, 10)
        self.convz2 = TransformerConv(10, 10)
        self.edge_encoder = Linear(6, 10)
        self.edge_encoder2 = Linear(10, 10)
        #self.convx1 = PDNConv(10, 10, edge_dim=10, hidden_channels=10)
        #self.convx2 = PDNConv(10, 10, edge_dim=10, hidden_channels=10)
        self.convx1 = TransformerConv(10, 10, edge_dim=10)
        self.convx2 = TransformerConv(10, 10, edge_dim=10)



        self.linxz = Linear(20, 10)
        self.convxz1 = GeneralConv(10, 10)
        self.convxz2 = GATv2Conv(10, 10)
        self.convxz3 = GATv2Conv(10, 10)
        self.convxz4 = GATv2Conv(10, 10)
        self.convxz4 = SAGEConv(10, 10, aggr = 'max')
        self.convxz5 = GATv2Conv(10, 10)
        self.linL = Linear(10, 10)
        self.convleaf = TransformerConv(10, 10, edge_dim = 10)
        #self.convleaf = PDNConv(10, 10, edge_dim = 10, hidden_channels = 10)
        #self.conv1 = GATCGATv2Convonv(-1, hidden_channels, edge_dim = dataset.num_edge_features)
        self.lin = Linear(10, 2)

    def forward(self, x, z, edge_index, z1edge_index, z2edge_index, z3edge_index,  edge_attr, pickable):
        # 1. Obtain node embeddings
        #y = self.edge_encoder(edge_attr)
        #x = x.relu()
        x = self.node_encx1(x)
        x = x.relu()
        z = self.node_encz1(z)
        #z = z.relu()

        edge_attr = self.edge_encoder(edge_attr)
        edge_attr = self.edge_encoder2(edge_attr)

        x = self.convx1(x, edge_index, edge_attr = edge_attr)
        x = x.relu()
        x = self.convx2(x, edge_index, edge_attr = edge_attr)

        z = self.convz1(z, z1edge_index)
        z = z.relu()
        z = self.convz2(z, z1edge_index)

        x = torch.hstack((x, z))
        x = self.linxz(x)
        x = self.convxz1(x,z1edge_index)
        x = x.relu()
        x = self.convxz2(x,z2edge_index)
        x = x.relu()
        x = self.convxz3(x,z1edge_index)
        x = x.relu()
        x = self.convxz4(x,z3edge_index)
        x = x.relu()
        x = self.convxz5(x,z1edge_index)
        x = x.relu()

        #x = self.convleaf(x, edge_index, edge_attr = edge_attr)

        x = self.lin(x)
        x = x[pickable]
        x = F.softmax(x,dim=1)
        #x = F.softmin(x,dim=1)

        # Normalize the features by dividing each element by the total


        return x

class netw(torch.nn.Module):
    def __init__(self):
        super(netw, self).__init__()
        # GATConv
        # TransformerConv
        # GATv2Conv
        self.node_encz1 = Linear(13, 10)
        self.convz1 = TransformerConv(10, 10)
        self.convz2 = TransformerConv(10, 10)

        self.linxz = Linear(10, 10)
        self.convxz1 = GeneralConv(10, 10)
        self.convxz2 = GATv2Conv(10, 10)
        self.convxz3 = GATv2Conv(10, 10)
        self.convxz3A = GATv2Conv(10, 10)
        self.convxz3B = GATv2Conv(10, 10)
        self.convxz4 = GATv2Conv(10, 10)
        self.convxz4 = SAGEConv(10, 10, aggr = 'max')
        self.convxz5 = GATv2Conv(10, 10)
        self.linL = Linear(10, 10)
        self.convleaf = TransformerConv(10, 10, edge_dim = 10)
        #self.convleaf = PDNConv(10, 10, edge_dim = 10, hidden_channels = 10)
        #self.conv1 = GATCGATv2Convonv(-1, hidden_channels, edge_dim = dataset.num_edge_features)
        self.lin = Linear(10, 2)

    def forward(self, x, z, edge_index, z1edge_index, z2edge_index, z3edge_index, z4edge_index, z5edge_index,  edge_attr, pickable):
        # 1. Obtain node embeddings
        #y = self.edge_encoder(edge_attr)
        #x = x.relu()
        z = self.node_encz1(z)
        #z = z.relu()



        z = self.convz1(z, z1edge_index)
        z = z.relu()
        z = self.convz2(z, z1edge_index)


        x = self.linxz(z)
        x = self.convxz1(x,z1edge_index)
        x = x.relu()
        x = self.convxz2(x,z2edge_index)
        x = x.relu()
        x = self.convxz3A(x,z4edge_index)
        #x = self.convxz3A(x,z1edge_index)
        #x = x.relu()
        #x = self.convxz3B(x,z5edge_index)
        #x = self.convxz3B(x,z1edge_index)
        x = x.relu()
        x = self.convxz3(x,z1edge_index)
        x = x.relu()
        x = self.convxz4(x,z3edge_index)
        x = x.relu()
        x = self.convxz5(x,z1edge_index)
        x = x.relu()

        #x = self.convleaf(x, edge_index, edge_attr = edge_attr)

        x = self.lin(x)
        x = x[pickable]
        x = F.softmax(x,dim=1)
        #x = F.softmin(x,dim=1)

        # Normalize the features by dividing each element by the total


        return x


class clique(torch.nn.Module):
    def __init__(self,xf,exf,out):
        super(clique, self).__init__()
        # GATConv
        # TransformerConv
        # GATv2Conv
        self.l0 = Linear(xf,xf)
        self.l0e = Linear(exf,exf)
        self.n2 = GraphNorm(exf)
        self.e1 = Linear(exf, 10)
        self.e2 = Linear(10, 10)
        self.eA = Linear(exf, 10)
        self.l1 = Linear(xf, 10)

        self.n = GraphNorm(xf)
        self.l2 = TransformerConv(10+xf, 10, edge_dim=10)
        self.l3 = TransformerConv(10+xf, 10, edge_dim=10)
        self.l4 = TransformerConv(10+xf, 10, edge_dim=10)
        self.l5 = TransformerConv(10+xf, 10, edge_dim=10)
        #self.convx4 = TransformerConv(10, 10, edge_dim=10)
        #self.convx1 = PDNConv(10, 10, edge_dim=10, hidden_channels=10)
        #self.convx2 = PDNConv(10, 10, edge_dim=10, hidden_channels=10)
        #self.convx3 = PDNConv(10, 10, edge_dim=10, hidden_channels=10)

        self.lin = Linear(10, out)

    def forward(self, x, z, edge_index, z1edge_index, z2edge_index, z3edge_index, z4edge_index, z5edge_index,  edge_attr, pickable):
        # 1. Obtain node embeddings
        x1 = x
        x1 = self.l0(x1)
        x = self.n(x1)
        x1 = x
        x1 = self.l1(x1)
        x1 = x1.relu()
        #z = z.relu()

        edge_attr = self.l0e(edge_attr)
        edge_attr = self.n2(edge_attr)
        edge_attrB = self.eA(edge_attr)
        edge_attr = self.e1(edge_attr)
        edge_attr = edge_attr.relu()
        edge_attr = self.e2(edge_attr)

        x1 = torch.hstack((x,x1))
        x1 = self.l2(x1, edge_index, edge_attr = edge_attr)
        x1 = x1.relu()
        x1 = torch.hstack((x,x1))
        x1 = self.l3(x1, edge_index, edge_attr = edge_attr)
        x1 = x1.relu()
        x1 = torch.hstack((x,x1))
        x1 = self.l4(x1, edge_index, edge_attr = edge_attrB)
        x1 = x1.relu()
        x1 = torch.hstack((x,x1))
        x1 = self.l5(x1, edge_index, edge_attr = edge_attrB)
        #x = x.relu()
        #x = self.convx4(x, edge_index, edge_attr = edge_attrB)


        x1 = self.lin(x1)
        x1 = x1[pickable]
        x1 = F.softmax(x1,dim=1)
        #x = F.softmin(x,dim=1)


        return x1


if __name__ == '__main__':
    def train():
        model.train()

        for data in train_loader:  # Iterate in batches over the training dataset.
            out = model(data.x, data.z, data.edge_index, data.z1edge_index, data.z2edge_index, data.z3edge_index, data.z4edge_index, data.z5edge_index,
                data.edge_attr, data.pickable) # Perform a single forward pass.
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
            out = model(data.x, data.z, data.edge_index, data.z1edge_index, data.z2edge_index, data.z3edge_index, data.z4edge_index, data.z5edge_index,
                data.edge_attr, data.pickable)
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

    dataset = MyOwnDataset('DataGen/ret_L15_T2', sample = 'all', delete = True)
    #dataset = MyOwnDataset('DataGen/ret_L15_T3', sample = 200, delete = False)
    #dataset = MyOwnDataset('DataGen/ret_lowret', sample = 200, delete = False)
    #dataset = MyOwnDataset('DataGen/ret_cyc', sample = 200, delete = True)
    #dataset = MyOwnDataset('DataGen/ret_lowret', sample = 200, delete = False)

    x_f = dataset[0].x.size()[1]
    ex_f = dataset[0].edge_attr.size()[1]
    num_classes = dataset.num_classes
    #model = GCN()
    model = clique(x_f,ex_f,num_classes)
    #model = netw()

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
    out = model(data.x, data.z, data.edge_index, data.z1edge_index, data.z2edge_index, data.z3edge_index, data.z4edge_index, data.z5edge_index,
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

    plotSet = [n[0]+n[1] for n in ratio_train]
    plt.plot(x, plotSet, linestyle='dotted', color=colours[i])

    print(max(np.sum(ratio_train, axis=1)))
    print(min(loss_data))
    plt.figure(1)
    plt.plot(x, loss_data)
    plt.show()
