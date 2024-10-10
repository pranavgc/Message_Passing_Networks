import torch
import numpy as np
import pandas as pd
from torch import nn
#from torch.utils.data import DataLoader
#from torchvision import datasets
#from torch.utils.data import Dataset
from torchvision.transforms import ToTensor, Lambda
import torch.nn.functional as F

from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch_geometric.data import Dataset
import torch_geometric.transforms as T
from torch_geometric.nn import GINEConv, BatchNorm, MessagePassing
from torch_geometric.nn import global_add_pool
from torch_geometric.utils import add_self_loops as sloops




s=pd.read_csv('data.txt'',sep=" ",usecols=columns) 

cell= np.array([s['2'],s['3'],s['4'],s['l']])
cell=np.transpose(cell)


class CustomDataset(Dataset):
    def __init__(self, labels, nodes,mark, edges, edge_a, transform=None, target_transform=None,add_self_loops: bool = True,prominent:bool=True,add_xy:bool=False):
    #def __init__(self, data_file, transform=None, target_transform=None):
        #self.labels = pd.read_csv(labels_file)
        #self.labels=labels.astype(int)
        self.labels=labels
        self.nodes=nodes
        self.mark=mark
        self.ei=edges
        self.ea=edge_a
        self.transform = transform
        self.ud=T.ToUndirected(reduce='max')
        self.target_transform = target_transform
        self.add_self_loops=add_self_loops
        self.prominent=prominent
        self.add_xy=add_xy
    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        data_n = self.nodes[idx][:,1:3]
        data_mark=list(self.mark[idx].values())
        
        
        
        data_ei=self.ei[idx]
        data_ea=self.ea[idx]
        label=self.labels[idx]
        
        'set x1 acc to data_n'
        x1=torch.tensor(data_n, dtype=torch.float32)
        #x1=torch.transpose(torch.tensor([data_n], dtype=torch.float32),0,1)
        #print("x1 shape",x1.shape)
        
        if self.add_xy:
            d_x,d_y=[],[]
            for i in self.nodes[idx][:,0]:
                d_x.append(cell[int(i),0])
                d_y.append(cell[int(i),1])
            d_x=torch.transpose(torch.tensor([d_x], dtype=torch.float32),0,1)
            d_y=torch.transpose(torch.tensor([d_y], dtype=torch.float32),0,1)
            x1=torch.cat((x1,d_x,d_y),dim=1)
        #print("x1 shape after xy",x1.shape)
        
        "Maximal or not"
        if self.prominent:
            d_mark=torch.reshape(torch.tensor(data_mark, dtype=torch.float32),(len(data_mark),1))
        #print(d_mark.shape)
        #print(torch.tensor(data_n, dtype=torch.float32))
        #x1=torch.tensor(data_n, dtype=torch.float32)
            x1= torch.cat((x1,d_mark),dim=1)
            
                             
                             
        x1=F.normalize(x1,p=2.0,dim=0)
        edge_ind1=torch.tensor(data_ei,dtype=torch.long)
        ei=edge_ind1.t().contiguous()
        #print("ei shape before self looping",ei.shape)
        ea=torch.tensor(data_ea,dtype=torch.float32)
        #ed1=F.normalize(ed1,p=2.0,dim=0)
        
        '''adding self loops'''
        if self.add_self_loops:
            a2,b2=sloops(ei,edge_attr=ea,num_nodes=len(x1[:,0]))
            b2[len(ei[0,:]):]=torch.reshape(x1[:,0],shape=(len(x1[:,0]),1))
        
            adj=torch.cat((a2,torch.transpose(b2,0,1)),dim=0)
            #print("Adj shape",adj.shape)
            #rint("ei/ea shape after self looping",a2.shape,b2.shape)
            #print(adj[2,:])
            
            adj=adj[:,adj[0, :].sort()[1]]
            #print(adj[2,:])
            #data = Data(x=x1, edge_index=adj[:2,:].type(torch.long), edge_attr=adj[2,:])
            ei=adj[:2,:].type(torch.long)
            ea=adj[2,:]
        
        label=torch.tensor(label,dtype=torch.long)
        #data = Data(x=x1, edge_index=edge_ind1.t().contiguous(), edge_attr=ed1)
        data = Data(x=x1, edge_index=ei, edge_attr=ea)
        
        #data=self.ud(data)
        data.y=label
        

        if self.transform is not None:
                data = self.pre_transform(data)
        
        return data
    


class GCN_v2(MessagePassing):
    def __init__(self,in_channels: int, out_channels: int):
        super().__init__(aggr='add')  # "Add" aggregation (Step 5).
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mlp=nn.Sequential(
            torch.nn.Linear(in_channels*2, out_channels),
            BatchNorm(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            
            torch.nn.Linear(out_channels,out_channels),
            BatchNorm(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),
            
            torch.nn.Linear(out_channels, out_channels,bias=False),
            BatchNorm(out_channels, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(),)
        
        #self.bias = torch.nn.Parameter(torch.Tensor(hidden_channels))
        #self.lin1 =torch.nn.Linear(in_channels*2, out_channels, bias=False)
        self.bias = torch.nn.Parameter(torch.Tensor(out_channels))
        #self.dropout=dropout

        self.reset_parameters()

    def reset_parameters(self):
        #self.lin1.reset_parameters()
        self.bias.data.zero_()

    def forward(self, x, edge_index,edge_attr):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        # Step 1: Add self-loops to the adjacency matrix.
        #edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2: Linearly transform node feature matrix.
        #x = self.lin(x)
        #x2=x[:,0]
        #x=x[:,1:]

        # Step 3: Compute normalization.
        target, neighbour = edge_index
        #deg = degree(col, x.size(0), dtype=x.dtype)
        #deg_inv_sqrt = deg.pow(-0.5)
        #deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        #norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        norm=x[target,0]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, edge_weight=edge_attr,norm=norm)

        # Step 6: Apply a final bias vector.
        out += self.bias

        return out

    def message(self, x_j, x_i,edge_weight,norm):
        # x_j has shape [E, out_channels]

        # Step 4: Normalize node features.
        #print("\nshape x_i",x_i.shape,"x_j",x_j.shape)
        in_f=torch.cat((x_i,x_i-x_j),dim=1)
        #in_f=torch.reshape(in_f,(1,len(in_f)))
        #in_f=x_j
        #print("\nshape input in_f",in_f.shape)
        out_f=self.mlp(in_f)
        #out_f=self.bns(out_f)
        #out_f=out_f.relu()
        #print("\nout_f shape bf edg_w",out_f.shape)
        #print("norm shape",norm.shape,"edge_attr shape",edge_weight.shape)
        out_f=norm.view(-1,1)*edge_weight.view(-1,1)*out_f
        #print("out_f shape af edg_w",out_f.shape)
        #out_f = F.dropout(out_f, p=self.dropout, training=self.training)
        
        
        return out_f 
    
    
    
class MP_GNN(torch.nn.Module):
    def __init__(self,input_channels, hidden_channels,num_classes,num_conv_layers,dropout, pool):
        super().__init__()
        torch.manual_seed(12345)
        self.num_conv_layers = num_conv_layers
        self.dropout = dropout
    
        #self.conv1 = GCN_v2(input_channels, hidden_channels,self.dropout)
        #self.conv2 = GCN_v2(hidden_channels, hidden_channels)
        #self.conv3 = GCNConv(hidden_channels, hidden_channels)
        #self.conv3 = GCNConv(hidden_channels, 32)
        
        self.convs = nn.ModuleList([GCN_v2(input_channels,hidden_channels)])
        self.convs.extend([GCN_v2(hidden_channels,hidden_channels) for i in range(num_conv_layers)])
        
        self.linear_stack = nn.Sequential(
            torch.nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            torch.nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_channels),
            torch.nn.Linear(hidden_channels, num_classes),
            nn.Softmax(dim=1),)
        self.pool = pool
        self.bns_f = nn.BatchNorm1d(hidden_channels)
        self.bns = nn.ModuleList([nn.BatchNorm1d(hidden_channels) for i in range(num_conv_layers)])
        self.lin0=torch.nn.Linear(input_channels, hidden_channels)
        

    def forward(self, x, edge_index,edge_attr, batch):
        # 1. Obtain node embeddings 
        #x=torch.reshape(x,(len(x),1))
        for i in range(self.num_conv_layers):
            if i!=self.num_conv_layers-1:
                x = self.convs[i](x, edge_index,edge_attr)
                x = self.bns[i](x)
                #edge_attr=nn.BatchNorm1d(len(edge_attr))
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
            else:
                x = self.convs[i](x, edge_index,edge_attr)

        # 2. Readout layer
        x = self.pool(x, batch)
        #print('pool batch',batch)# [batch_size, hidden_channels]
        #print('pool' , x.shape)

        # 3. Apply a final classifier
        x = F.dropout(x, p=self.dropout, training=self.training)
        x=self.bns_f(x)
        out = self.linear_stack(x)
        #print('prediction',out.shape)
        
        return out
def train_loop(dataloader, model, loss_fn, optimizer,dev):
    model.train()
    size = len(dataloader.dataset)
    i_train=0
    for step,data in enumerate(dataloader):# Iterate in batches over the training dataset.
        print("\n\n==================================================================\n\n")
        print('train loop counter',i_train)
        print("\n\n-------------------------------------------------------------------\n\n")
        data=data.to(dev)
        out = model(data.x, data.edge_index,data.edge_attr, data.batch)# Perform a single forward pass.
        #print('In train before loss out shape',out.shape, 'y shape',data.y.shape)

        loss = loss_fn(out, data.y)  # Compute the loss.

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        i_train += 1


        '''if step % 100 == 0:
            loss, current = loss.item(), step* len(data)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")'''
        loss = loss.item()
        print(f"loss: {loss:>7f}")

def test_loop(dataloader, model, loss_fn,dev):
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0
    c=np.zeros(1)
    c1=np.zeros(1)
    with torch.no_grad():
        for data in dataloader:
            data=data.to(dev)# Iterate in batches over the training/test dataset.
            pred = model(data.x, data.edge_index,data.edge_attr, data.batch)
            
            #pred = out.argmax(dim=1)  # Use the class with highest probability.
            pred2=pred.cpu()
            #print('test pred shape',torch.argmax(pred2, dim=1).shape)
            #time.sleep(5)
            c=np.append(c,torch.argmax(pred2, dim=1))
            
            test_loss += loss_fn(pred, data.y).item()
            correct += (pred.argmax(1) == data.y).type(torch.float).sum().item()
            y1=data.y.detach().clone()
            y1=y1.cpu()
            c1=np.append(c1,y1)

    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.2f}%, Avg loss: {test_loss:>8f} \n")
    #return c,c1,correct,jac1_te
    return c,c1,correct