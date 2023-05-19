import argparse
import json
import dgl
from sklearn.metrics import f1_score
import torch as th
import dgl.nn.pytorch as dglnn
import torch.nn as nn
from dgl.dataloading import GraphDataLoader
import torch.nn.functional as F
import networkx as nx
import warnings
from feature_extraction.base_featurizer import BaseFeaturizer
import torch
from graph_construction.tps_graph import create_dummy_dgl_graph, tps_graph
import numpy as np

import warnings

from preprocessing.utils import load_BGPS_from_json
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

dgl.seed(1223)


def snap_pred(pred):
    if isinstance(pred, int):
        if pred >= 0.5:
            return 1
        else:
            return 0
    return torch.argmax( pred)

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
        self.conv1 = dglnn.RelGraphConv(in_dim, hidden_dim, 8)
        #self.conv1 = dglnn.GraphConv(in_dim, hidden_ˇdim)
        self.conv2 = dglnn.RelGraphConv(hidden_dim, hidden_dim, 8)
        #self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g, h, rel_types):
        # Apply graph convolution and activation.
        h = F.relu(self.conv1(g, h, rel_types))
        h = F.relu(self.conv2(g, h, rel_types))
        with g.local_scope():
            g.ndata['node_features'] = h
            # Calculate graph representation by average readout.
            hg = dgl.mean_nodes(g, 'node_features')
            return F.softmax( self.classify(hg), dim=1)
def tps_to_dgl(tps):
    """for n in tps.graph.nodes(data=True):
        print(n)
        exit()"""
    dgl_graph =  dgl.from_networkx(tps.graph, edge_attrs=['rel_type'], node_attrs=['node_features'])
    dgl_graph = dgl.add_self_loop( dgl_graph)
    return dgl_graph

def get_clasification_vec(gt):
    vec =np.zeros(2)
    vec[gt] = 1
    return vec
def tps_graph_const(x, featurizer):
    try:
        i = tps_graph(x, featurizer=featurizer)
    except AssertionError:
        return None
    return i
def extract_data(train_path, val_path, test_path, community_no=10, batch_size = 50, verbose=False):
    #featurizer = BaseFeaturizer( train_log = train_path, is_pred_clust_feat=True,save_pred_graph_png = None, 
    #                            community_no=community_no,path_pred_clust = { 'save_path':'/work/data/confs/May2/pred_clust.json', 'load_path':None}, verbose=False)
    #featurizer = BaseFeaturizer( train_log = train_path, is_pred_clust_feat=True,save_pred_graph_png = 'pred_graph.html', 
    #                            community_no=community_no,path_pred_clust = { 'save_path':None, 'load_path':None}, verbose=False)
    featurizer = BaseFeaturizer( train_log = train_path, is_pred_clust_feat=True,save_pred_graph_png = None, community_no=community_no,path_pred_clust = { 'save_path':None, 'load_path':'/work/data/confs/May2/pred_clust.json'}, verbose=False)
    
    train_bgps = load_BGPS_from_json(train_path)
    #train_dgl = [tps_to_dgl( tps_graph_const(x, featurizer=featurizer)) for x in train_bgps if not tps_graph_const(x, featurizer=featurizer) == None]
    train_dgl,train_gt = [], []
    for x in train_bgps:
        dgl_graph = tps_graph_const(x, featurizer=featurizer)
        if dgl_graph != None:
            train_dgl.append(tps_to_dgl(dgl_graph))
            train_gt.append(get_clasification_vec(x.ground_truth))
    train_gt = th.tensor(train_gt, dtype=th.float32)
    #train_gt = th.tensor([get_clasification_vec(x.ground_truth) for x in train_bgps], dtype=th.float32)
    
    val_bgps = load_BGPS_from_json(val_path)
    #val_dgl = [tps_to_dgl(tps_graph_const(x, featurizer=featurizer)) for x in val_bgps if not tps_graph_const(x, featurizer=featurizer) == None]
    val_dgl,val_gt = [], []
    for x in val_bgps:
        dgl_graph = tps_graph_const(x, featurizer=featurizer)
        if dgl_graph != None:
            val_dgl.append(tps_to_dgl(dgl_graph))
            val_gt.append(get_clasification_vec(x.ground_truth))
    val_gt = th.tensor(val_gt, dtype=th.float32)
    
    test_bgps = load_BGPS_from_json(test_path)
    #test_dgl = [tps_to_dgl(tps_graph_const(x, featurizer=featurizer)) for x in test_bgps if not tps_graph_const(x, featurizer=featurizer) == None]
    test_dgl,test_gt = [], []
    for x in test_bgps:
        dgl_graph = tps_graph_const(x, featurizer=featurizer)
        if dgl_graph != None:
            test_dgl.append(tps_to_dgl(dgl_graph))
            test_gt.append(get_clasification_vec(x.ground_truth))
    test_gt = th.tensor(test_gt, dtype=th.float32)
    
    #gt regression typ
    """train_gt = th.tensor([[x.ground_truth] for x in train_bgps], dtype=th.float32)
    val_gt = th.tensor([[x.ground_truth] for x in val_bgps], dtype=th.float32)
    test_gt = th.tensor([[x.ground_truth] for x in test_bgps], dtype=th.float32)"""
    #gt classification type
    #train_gt = th.tensor([get_clasification_vec(x.ground_truth) for x in train_bgps], dtype=th.float32)
    #val_gt = th.tensor([get_clasification_vec(x.ground_truth) for x in val_bgps], dtype=th.float32)
    #test_gt = th.tensor([get_clasification_vec(x.ground_truth) for x in test_bgps], dtype=th.float32)
    
    train_dataset = GraphDataset(train_dgl, train_gt)
    train_dataloader = GraphDataLoader(
        train_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=True)
    """for batched_graph, labels in train_dataloader:
        print(labels)
    exit()"""
    val_dataset = GraphDataset(val_dgl, val_gt)
    val_dataloader = GraphDataLoader(
        val_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=True)
    test_dataset = GraphDataset(test_dgl, test_gt)
    test_dataloader = GraphDataLoader(
        test_dataset,
        batch_size=batch_size,
        drop_last=False,
        shuffle=True)
    return train_dataloader, val_dataloader, test_dataloader

if __name__ == "__main__":
    parser = argparse.ArgumentParser('DGL trainer')
    parser.add_argument('--train_file','--train_file')
    parser.add_argument('--val_file','--val_file')
    parser.add_argument('--test_file','--test_file')
    args = parser.parse_args()
    train_file = args.train_file
    val_file = args.val_file
    test_file = args.test_file
    train_dataloader, val_dataloader, test_dataloader = extract_data(train_file, val_file, test_file, community_no=30, batch_size = 50, verbose=False)
    
    path_to_save = '/work/data/models'
    
    
    early_stop =10
    model = Classifier(113, 20, 2)
    lr = 0.01
    wd = 5e-4
    opt = th.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    
    prev_val_loss = None
    val_hist = []
    best_epoch = 0
    for epoch in range(100):
        print(f"Epoch {epoch+1}\n--------------------------------------------------------------")

        train_loss = 0
        train_f1 = 0
        model.train()
        
        for batch_no,(batched_graph, labels) in enumerate(train_dataloader):
            feats = batched_graph.ndata['node_features']
            edge_types = batched_graph.edata['rel_type']
            logits = model(batched_graph, feats, edge_types)
            loss = F.cross_entropy(logits, labels)
            #loss = F.mse_loss(logits,labels)
            #cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            #print(f"{labels}, {logits}")
            opt.step()
            
            c_train_loss = loss.item()
            f1_pred = list(map(snap_pred,logits))
            snapped_labels = list(map(snap_pred,labels))
            #f1_batch = f1_score(labels, f1_pred)
            f1_batch = f1_score(snapped_labels, f1_pred)
            print(f"Epoch: {epoch+1:4} {(batch_no+1):8} Batch loss: {c_train_loss:>7f} Batch F1: {f1_batch}")
            train_loss += c_train_loss
            train_f1 += f1_batch
        
        val_loss= 0
        val_f1 = 0
        model.eval()
        for no,(graphs, labels) in enumerate(val_dataloader):
            feats = graphs.ndata['node_features']
            edge_types = graphs.edata['rel_type']
            pred = model(graphs, feats, edge_types)
            c_test_loss = F.mse_loss(pred, labels).item()
            #test_loss += loss_fn(pred, torch.reshape(sample['target'],(1,1))).item()
            val_loss += c_test_loss
            f1_pred_val = list(map(snap_pred,pred))
            snapped_lebls = list(map(snap_pred,labels))
            
            #f1_batch_val = f1_score(labels, f1_pred_val)
            f1_batch_val = f1_score(snapped_lebls, f1_pred_val)
            val_f1 += f1_batch_val
        val_loss= val_loss/len(val_dataloader)
        val_f1= val_f1/len(val_dataloader)
        train_loss = train_loss/len(train_dataloader)
        train_f1 = train_f1/len(train_dataloader)
    
        p_val_hist = val_hist
        val_hist.append(val_loss)
        if path_to_save != None and (prev_val_loss == None or val_loss < prev_val_loss):
            torch.save(model,f"{path_to_save}/model_{epoch+1}.pt")
            prev_val_loss = val_loss
            best_epoch = epoch+1
        
        print(f"Train Avg Loss {epoch+1:4}: {train_loss:>8f}\n")
        print(f"Train Avg F1 {epoch+1:4}: {train_f1}\n")
        print(f"Val Avg Loss {epoch+1:4}: {val_loss:>8f}\n")
        print(f"Val Avg F1 {epoch+1:4}:  {val_f1}\n")
        print(f"Optimal Val loss (Epoch {best_epoch}): {prev_val_loss}\n")
        
        if (early_stop < len(p_val_hist)) and (np.sum([1 for v_l in p_val_hist[-early_stop:] if v_l <= val_loss]) == early_stop):
            print(f"Early Stopping invoked after epoch {epoch+1}")
            exit()
    print("Done!")
    
    #test
    exit()
    g = create_dummy_dgl_graph()
    
    dataset = GraphDataset([g,g], th.tensor( [[1.0],[1.0]]))
    dataloader = GraphDataLoader(
        dataset,
        batch_size=2,
        drop_last=False,
        shuffle=True)
    model = Classifier(66, 20, 1)
    opt = th.optim.Adam(model.parameters())
    for epoch in range(20):
        for batched_graph, labels in dataloader:
            feats = batched_graph.ndata['node_features']
            edge_types = batched_graph.edata['rel_type']
            logits = model(batched_graph, feats, edge_types)
            loss = F.cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()