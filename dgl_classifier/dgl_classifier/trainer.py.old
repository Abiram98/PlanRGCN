import dgl
import torch as th
import dgl.nn.pytorch as dglnn
import torch.nn as nn
from dgl.dataloading import GraphDataLoader
import torch.nn.functional as F
import networkx as nx
import warnings
warnings.filterwarnings('ignore')
dgl.seed(1223)

graph = nx.Graph()
nodes = [2,4,3,5]
node_feat = [[1,2] for x in nodes]
graph.add_nodes_from(nodes, feat1=node_feat)
graph.add_edge(2,4)
graph.add_edge(3,4)
graph.add_edge(5,4)

g1 = dgl.from_networkx(graph, node_attrs=['feat1'])


g1 = dgl.graph((th.tensor([0, 1, 2]), th.tensor([1, 2, 3])))
g1 = dgl.add_self_loop(g1)
g1.ndata['h'] = th.tensor([[1.,2.], [2.,2.],[4.,2.], [5.,2.]])
g2 = dgl.graph((th.tensor([0, 0, 0, 1]), th.tensor([0, 1, 2, 0])))
g2.ndata['h'] = th.tensor([[1.,2.], [2.,2.], [3.,2.]])
g2 = dgl.add_self_loop(g2)

class GraphDataset:
    def __init__(self, graphs, labels) -> None:
        self.graph = graphs
        self.labels = labels
    
    def __getitem__(self, i):
        return self.graph[i], self.labels[i]
    
    def __len__(self):
        return len(self.labels)

class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'h')
            return self.classify(hg)

dataset = GraphDataset([g1,g2], th.tensor( [[1.0],[2.0]]))
dataloader = GraphDataLoader(
    dataset,
    batch_size=2,
    drop_last=False,
    shuffle=True)
    
model = Classifier(2, 20, 1)
opt = th.optim.Adam(model.parameters())
for epoch in range(20):
    for batched_graph, labels in dataloader:
        feats = batched_graph.ndata['h']
        logits = model(batched_graph, feats)
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()