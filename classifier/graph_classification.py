from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv, SAGEConv
from torch_geometric.nn import global_mean_pool
import torch, numpy as np



class GNN(torch.nn.Module):
    def __init__(self, input_size, hidden_channels, no_embeddings=None, hidden_dimension=50):
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
    
    
    def forward(self, x, edge_index, join_index, batch = None,  edge_col = None):

        # Node embedding
        #x = self.IRIs(x)
        x = self.conv1(x, edge_index, edge_col)
        x = torch.relu(x)
        #x = x.relu()
        x = self.conv2(x, edge_index, edge_col)
        
        # Readout layer
        #batch = torch.zeros(data.x.shape[0],dtype=int) if batch is None else batch
        #x = global_mean_pool(x, batch)
        x = global_mean_pool(x,batch)
        # Final classifier
        #x = F.dropout(x, p=0.5, training=self.training)
        #maybe sigmoid here
        
        x = self.lin(x)

        x = torch.sigmoid(x)
        return x