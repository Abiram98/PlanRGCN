import dgl
import dgl.nn.pytorch as dglnn
from graph_construction.query_graph import QueryPlan, QueryPlanCommonBi
import torch.nn as nn
import torch.nn.functional as F
import torch


class Classifier(nn.Module):
    """_summary_
    Note: Data loading needs to happen before model construction as there is a dependency on the QuerPlan.max_relations
    Args:
        nn (_type_): _description_
    """

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


class Classifier2RGCN(nn.Module):
    """_summary_
    Note: Data loading needs to happen before model construction as there is a dependency on the QuerPlan.max_relations
    Args:
        nn (_type_): _description_
    """

    def __init__(self, in_dim, hidden_dim1, hidden_dim2, n_classes):
        super(Classifier2RGCN, self).__init__()
        self.conv1 = dglnn.RelGraphConv(in_dim, hidden_dim1, QueryPlan.max_relations)
        # self.conv1 = dglnn.GraphConv(in_dim, hidden_ˇdim)
        self.conv2 = dglnn.RelGraphConv(
            hidden_dim1, hidden_dim2, QueryPlan.max_relations
        )
        # self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim2, n_classes)
        self.in_dim = in_dim
        self.hidden_dim1 = hidden_dim1
        self.hidden_dim2 = hidden_dim2
        self.n_classes = n_classes

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

    def get_dims(self):
        return self.in_dim, self.hidden_dim1, self.hidden_dim2, self.n_classes


class ClassifierGridSearch(nn.Module):
    """_summary_
    Note: Data loading needs to happen before model construction as there is a dependency on the QuerPlan.max_relations
    Args:
        nn (_type_): _description_
    """

    def __init__(self, in_dim, layers, n_classes):
        super().__init__()
        self.rgcn_neurons = {}
        self.fc_neurons = {}
        self.layers = nn.ModuleList()
        prev = in_dim
        self.rgcn_last_idx = 0
        self.contains_fc = False
        for layer_no, (t, neurons) in enumerate(layers):
            if t == 1:
                self.layers.append(
                    dglnn.RelGraphConv(prev, neurons, QueryPlan.max_relations)
                )
                self.rgcn_neurons[f"RGCN {layer_no+1}"] = neurons
                self.rgcn_last_idx = layer_no
            elif t == 2:
                self.contains_fc = True
                # fully connected layers (optional)
                self.layers.append(nn.Linear(prev, neurons))
                self.fc_neurons[f"RGCN {layer_no+1}"] = neurons

            prev = neurons

        # self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(prev, n_classes)
        self.in_dim = in_dim
        self.n_classes = n_classes

    def forward(self, g, h, rel_types):
        if h.dtype != torch.float32:
            h = h.type(torch.float32)
        for layer_no, l in self.layers:
            if layer_no <= self.rgcn_last_idx:
                h = l(g, h, rel_types)
                h = F.relu(h)
            elif layer_no == (self.rgcn_last_idx + 1) and self.contains_fc:
                with g.local_scope():
                    g.ndata["node_features"] = h
                    # Calculate graph representation by average readout.
                    h = dgl.mean_nodes(g, "node_features")
                    h = F.relu(l(h))
            elif layer_no > (self.rgcn_last_idx + 1) and self.contains_fc:
                h = F.relu(l(h))
        if not self.contains_fc:
            with g.local_scope():
                g.ndata["node_features"] = h
                # Calculate graph representation by average readout.
                hg = dgl.mean_nodes(g, "node_features")
                return F.softmax(self.classify(hg), dim=1)
        else:
            return F.softmax(self.classify(h), dim=1)

    def get_dims(self):
        return self.in_dim, self.rgcn_neurons, self.fc_neurons, self.n_classes


class ClassifierWAuto(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(ClassifierWAuto, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, dtype=torch.float32), nn.ReLU()
        )
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, in_dim, dtype=torch.float32))
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
            return decoded, F.softmax(self.classify(hg), dim=1)


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
