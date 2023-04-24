from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import global_mean_pool
import torch, numpy as np

from torch_geometric.data import Data

class GNN(torch.nn.Module):
    def __init__(self, input_size, hidden_channels):
        super(GNN, self).__init__()
        self.add_layers(input_size,hidden_channels)
    
    #Graphsage layers
    def add_layers(self, input_size,hidden_channels):
        self.conv1 = SAGEConv(
            input_size, hidden_channels, normalize=True)
        self.conv1 = self.conv1.float()
        #self.norm1 = 
        self.conv2 = SAGEConv(
            hidden_channels, hidden_channels*2, normalize=True)
        self.conv2 = self.conv2.float()
        
        self.lin = Linear(hidden_channels*2, 1)
        self.lin = self.lin.float()
    
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x,batch)
        x = self.lin(x)
        x = torch.sigmoid(x)
        return x

class GNN_w_Dense(torch.nn.Module):
    def __init__(self, input_size, hidden_channels):
        super(GNN_w_Dense, self).__init__()
        self.add_layers(input_size,hidden_channels)
    
    #Graphsage layers
    def add_layers(self, input_size,hidden_channels):
        self.lin1 = Linear(input_size, hidden_channels)
        self.lin1 = self.lin1.float()
        self.conv1 = SAGEConv(
            hidden_channels, hidden_channels, normalize=True)
        self.conv1 = self.conv1.float()
        #self.norm1 = 
        self.conv2 = SAGEConv(
            hidden_channels, hidden_channels*2, normalize=True)
        self.conv2 = self.conv2.float()
        
        self.lin = Linear(hidden_channels*2, 1)
        self.lin = self.lin.float()
    
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.lin1(x)
        x = self.conv1(x, edge_index)
        x = torch.relu(x)
        x = self.conv2(x, edge_index)
        x = global_mean_pool(x,batch)
        x = self.lin(x)
        x = torch.sigmoid(x)
        return x

#for testing of models
if __name__ == "__main__":
    edge_index = torch.tensor([[0, 1, 1, 2],
                           [1, 0, 2, 1]], dtype=torch.long)
    x = torch.tensor([[-1,2,3], [0,2,3], [1,2,3]], dtype=torch.float)
    data = Data(x =x, edge_index=edge_index)
    model = GNN_w_Dense(3,10)
    model(data)