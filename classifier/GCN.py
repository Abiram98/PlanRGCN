#from https://blog.dataiku.com/graph-neural-networks-part-three 

from torch.nn import Linear
import torch.nn.functional as F
import torch_geometric
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool
import torch, numpy as np



class GNN(torch.nn.Module):
    def __init__(self, input_size, hidden_channels, no_embeddings=None, hidden_dimension=50):
        super(GNN, self).__init__()
        self.add_layers(input_size,hidden_channels)
        
    def add_layers(self, input_size,hidden_channels):
        self.conv1 = GCNConv(
            input_size, hidden_channels)
        
        self.conv2 = GCNConv(
            hidden_channels, hidden_channels*2)
        
        self.lin = Linear(hidden_channels*2, 1)
    
    
    def forward(self, x, edge_index, join_index, batch = None,  edge_col = None):

        # Node embedding
        #x = self.IRIs(x)
        x = self.conv1(x, edge_index, edge_col)
        
        x = x.relu()
        x = self.conv2(x, edge_index, edge_col)
        
        # Readout layer
        #batch = torch.zeros(data.x.shape[0],dtype=int) if batch is None else batch
        #x = global_mean_pool(x, batch)
        
        # Final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        #maybe sigmoid here
        x = self.lin(x[join_index])

        x = torch.sigmoid(x)
    
        return x
# data = [ (node_features, edgelist, joinindex, gt)]
def train_loop(data):
    torch.manual_seed(12345)
    model = GNN(10,10,hidden_dimension=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.MSELoss()
    model.train()
    EPOCHS = 100
    for i in range(EPOCHS):
        running_loss = 0.0
        for nodes, edge_indices, join_index,actual in data:
            optimizer.zero_grad()
            pred = model(nodes,edge_indices, join_index)
            loss = criterion(pred,actual)
            
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"Epoch: {i+1:05}: {running_loss:.3E}")
    
if __name__ == "__main__":
    nodes = [np.array([x]*10).astype(np.float32) for x in range(10)]
    
    nodes = torch.as_tensor(np.array(nodes))
    edge_indices = torch.as_tensor([[1,3,5,7,1],[2,4,6,8,8]])
    join_index = 0
    data = [(nodes,edge_indices,join_index, torch.as_tensor([1.0]))]
    #
    train_loop(data)
    #edge_indices = {[[1,3,5,7,1],[2,4,6,8,8]]}
    edge_indices = torch.as_tensor(np.array(edge_indices))
    model = GNN(10,10,hidden_dimension=10)
    
    #print(nodes.shape)
    model.eval()
    print(model(nodes,edge_indices, 0).item())
    #print(edge_indices.shape)
    #traced_cell = torch.jit.trace(model,(nodes,edge_indices))
    #traced_cell.save('test.pt')