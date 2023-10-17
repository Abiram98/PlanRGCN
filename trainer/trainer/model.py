import dgl
import dgl.nn.pytorch as dglnn
from graph_construction.query_graph import QueryPlan, QueryPlanCommonBi
import torch.nn as nn
import torch.nn.functional as F
import torch


class Classifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(Classifier, self).__init__()
        self.conv1 = dglnn.RelGraphConv(in_dim, hidden_dim, QueryPlan.max_relations)
        # self.conv1 = dglnn.GraphConv(in_dim, hidden_ˇdim)
        self.conv2 = dglnn.RelGraphConv(hidden_dim, hidden_dim, QueryPlan.max_relations)
        # self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h, rel_types):
        # Apply graph convolution and activation.
        if h.dtype != torch.float32:
            h = h.type(torch.float32)
        h = self.conv1(g, h, rel_types)
        h = F.relu(h)
        h = F.relu(self.conv2(g, h, rel_types))
        with g.local_scope():
            g.ndata["node_features"] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, "node_features")
            return F.softmax(self.classify(hg), dim=1)

class ClassifierWAuto(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(ClassifierWAuto, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, dtype=torch.float32),
            nn.ReLU()                 
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, in_dim, dtype=torch.float32)
        )
        self.dropout1 = nn.Dropout(0.2)
        self.conv1 = dglnn.RelGraphConv(hidden_dim, hidden_dim, QueryPlan.max_relations)
        # self.conv1 = dglnn.GraphConv(in_dim, hidden_ˇdim)
        self.dropout2 = nn.Dropout(0.2)
        self.conv2 = dglnn.RelGraphConv(hidden_dim, hidden_dim, QueryPlan.max_relations)
        # self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h, rel_types):
        # Apply graph convolution and activation.
        if h.dtype != torch.float32:
            h = h.type(torch.float32)
        h = self.encoder(h)
        decoded = self.decoder(h)
        h = self.dropout1(h)
        h = self.conv1(g, h, rel_types)
        h = F.relu(h)
        h = F.relu(self.conv2(g, h, rel_types))
        h = self.dropout2(h)
        with g.local_scope():
            g.ndata["node_features"] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, "node_features")
            return decoded,F.softmax(self.classify(hg), dim=1)


class ClassifierWSelfTriple(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(ClassifierWSelfTriple, self).__init__()
        self.conv1 = dglnn.RelGraphConv(
            in_dim, hidden_dim, QueryPlanCommonBi.max_relations
        )
        # self.conv1 = dglnn.GraphConv(in_dim, hidden_ˇdim)
        self.conv2 = dglnn.RelGraphConv(
            hidden_dim, hidden_dim, QueryPlanCommonBi.max_relations
        )
        # self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h, rel_types):
        # Apply graph convolution and activation.
        if h.dtype != torch.float32:
            h = h.type(torch.float32)
        h = self.conv1(g, h, rel_types)
        h = F.relu(h)
        h = F.relu(self.conv2(g, h, rel_types))
        with g.local_scope():
            g.ndata["node_features"] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, "node_features")
            return F.softmax(self.classify(hg), dim=1)


class RegressorWSelfTriple(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(RegressorWSelfTriple, self).__init__()
        self.conv1 = dglnn.RelGraphConv(
            in_dim, hidden_dim, QueryPlanCommonBi.max_relations
        )
        # self.conv1 = dglnn.GraphConv(in_dim, hidden_ˇdim)
        self.conv2 = dglnn.RelGraphConv(
            hidden_dim, hidden_dim, QueryPlanCommonBi.max_relations
        )
        # self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, 1)

    def forward(self, g, h, rel_types):
        # Apply graph convolution and activation.
        if h.dtype != torch.float32:
            h = h.type(torch.float32)
        h = self.conv1(g, h, rel_types)
        h = F.relu(h)
        h = F.relu(self.conv2(g, h, rel_types))
        with g.local_scope():
            g.ndata["node_features"] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, "node_features")
            return F.relu(self.classify(hg))
