import argparse
import json
import dgl
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
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
import numpy as np, os
import time
import warnings
from preprocessing.utils import load_BGPS_from_json
warnings.filterwarnings('ignore', category=UserWarning, message='TypedStorage is deprecated')

dgl.seed(1223)

def snap_pred(pred,  model_thres, add_thres):
    '''1. prediction, 2. model threshold, 3. bool whether to use threshold'''
    if isinstance(pred, int):
        if pred >= model_thres:
            return 1
        else:
            return 0
    if add_thres != None:
        # [0 , 1] = true /use bloom filters
        assert model_thres != None
        pred = pred.flatten()
        t = torch.argmax( pred)
        #assert pred.shape[0] == 2
        if (t ==1) and pred[1] > model_thres:
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
    dgl_graph =  dgl.from_networkx(tps.graph, edge_attrs=['rel_type'], node_attrs=['node_features'])
    dgl_graph = dgl.add_self_loop( dgl_graph)
    return dgl_graph

def get_clasification_vec(gt):
    vec =np.zeros(2)
    vec[gt] = 1
    return vec
def tps_graph_const(x, featurizer):
    try:
        start = time.time()
        i = tps_graph(x, featurizer=featurizer)
        duration = time.time()-start
        i.bgp.data_dict['tps_const_duration'] = duration
    except AssertionError:
        return None
    return i
def extract_data(train_path, val_path, test_path, community_no=10, batch_size = 50, verbose=True, clust_verbose=False):
    #featurizer = BaseFeaturizer( train_log = train_path, is_pred_clust_feat=True,save_pred_graph_png = None, 
    #                            community_no=community_no,path_pred_clust = { 'save_path':'/work/data/confs/May2/pred_clust.json', 'load_path':None}, verbose=False)
    #featurizer = BaseFeaturizer( train_log = train_path, is_pred_clust_feat=True,save_pred_graph_png = 'pred_graph.html', 
    #                            community_no=community_no,path_pred_clust = { 'save_path':None, 'load_path':None}, verbose=False)
    featurizer = BaseFeaturizer( train_log = train_path, is_pred_clust_feat=True,save_pred_graph_png = None, community_no=community_no,path_pred_clust = { 'save_path':None, 'load_path':'/work/data/confs/May2/pred_clust.json'}, verbose=clust_verbose)
    
    train_bgps = load_BGPS_from_json(train_path)
    #train_dgl = [tps_to_dgl( tps_graph_const(x, featurizer=featurizer)) for x in train_bgps if not tps_graph_const(x, featurizer=featurizer) == None]
    len_train = len(train_bgps)
    train_dgl,train_gt = [], []
    for x in train_bgps:
        dgl_graph = tps_graph_const(x, featurizer=featurizer)
        if dgl_graph != None:
            train_dgl.append(tps_to_dgl(dgl_graph))
            train_gt.append(get_clasification_vec(x.ground_truth))
    train_samples = len(train_dgl)
    train_gt = th.tensor(train_gt, dtype=th.float32)
    
    if verbose:
        print(f"# of Removed: {len_train-train_samples}")
    #train_gt = th.tensor([get_clasification_vec(x.ground_truth) for x in train_bgps], dtype=th.float32)
    
    val_bgps = load_BGPS_from_json(val_path)
    len_val = len(val_bgps)
    #val_dgl = [tps_to_dgl(tps_graph_const(x, featurizer=featurizer)) for x in val_bgps if not tps_graph_const(x, featurizer=featurizer) == None]
    val_dgl,val_gt = [], []
    for x in val_bgps:
        dgl_graph = tps_graph_const(x, featurizer=featurizer)
        if dgl_graph != None:
            val_dgl.append(tps_to_dgl(dgl_graph))
            val_gt.append(get_clasification_vec(x.ground_truth))
    val_gt = th.tensor(val_gt, dtype=th.float32)
    val_samples = len(val_dgl)
    val_pred_not_train = BaseFeaturizer.samples_not_in_train_log
    if verbose:
        print(f"# of Removed: {len_val-val_samples}")
    
    test_bgps = load_BGPS_from_json(test_path)
    len_test = len(test_bgps)
    #test_dgl = [tps_to_dgl(tps_graph_const(x, featurizer=featurizer)) for x in test_bgps if not tps_graph_const(x, featurizer=featurizer) == None]
    test_dgl,test_gt = [], []
    for x in test_bgps:
        dgl_graph = tps_graph_const(x, featurizer=featurizer)
        if dgl_graph != None:
            test_dgl.append(tps_to_dgl(dgl_graph))
            test_gt.append(get_clasification_vec(x.ground_truth))
    test_gt = th.tensor(test_gt, dtype=th.float32)
    test_samples = len(test_dgl)
    test_pred_not_train = BaseFeaturizer.samples_not_in_train_log - val_pred_not_train
    if verbose:
        print(f"# of Removed: {len_test-test_samples}")
    
    if verbose:
        print(f"# of predicate feats not in validation {val_pred_not_train}")
        print(f"# of predicate feats not in test {test_pred_not_train}")
        
        print(f"Training dataset size {train_samples}")
        print(f"Validation dataset size {val_samples}")
        print(f"Test dataset size {test_samples}")
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


def parse_arguments():
    parser = argparse.ArgumentParser('DGL trainer')
    parser.add_argument('--train_file','--train_file')
    parser.add_argument('--val_file','--val_file')
    parser.add_argument('--test_file','--test_file')
    parser.add_argument('--result_path','--result_path')
    args = parser.parse_args()
    return args

#runner for notebooks
def runner(train_dataloader,val_dataloader, test_dataloader,model, early_stop, lr, wd, epochs, result_path, path_to_save='/work/data/models', loss_type='cross-entropy',
           add_thres = False, pred_thres=0.5,
           verbosity=0):
    
    opt = th.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    #opt = th.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    
    prev_val_loss = None
    val_hist = []
    
    #F1 scores
    train_f1_hist = []
    val_f1_hist = []
    metric_data = {'train_loss':[],
                    'val_loss':[],
                    'test_loss':[],
                    'train_f1':[],
                    'val_f1':[],
                    'test_f1':[],
                    'train_prec':[],
                    'val_prec':[],
                    'test_prec':[],
                    'train_recall':[],
                    'val_recall':[],
                    'test_recall':[],
                    }
    best_epoch = 0
    best_f1 = 0
    best_model_path = ''
    for epoch in range(epochs):
        if verbosity >=2:
            print(f"Epoch {epoch+1}\n--------------------------------------------------------------")

        train_loss = 0
        train_f1 = 0
        train_recall= 0
        train_prec = 0
        model.train()
        
        for batch_no,(batched_graph, labels) in enumerate(train_dataloader):
            feats = batched_graph.ndata['node_features']
            edge_types = batched_graph.edata['rel_type']
            logits = model(batched_graph, feats, edge_types)
            if loss_type == 'cross-entropy':
                loss = F.cross_entropy(logits, labels)
            elif loss_type=='mse':
                loss = F.mse_loss(logits, labels)
            #loss = F.mse_loss(logits,labels)
            #cross_entropy(logits, labels)
            opt.zero_grad()
            loss.backward()
            #print(f"{labels}, {logits}")
            opt.step()
            
            c_train_loss = loss.item()
            
            #snap_pred(logits,model_thres=pred_thres, add_thres=add_thres)
            snap_thres = [pred_thres for x in logits]
            snap_add_thres = [add_thres for x in logits]
            f1_pred = list(map(snap_pred,logits,snap_thres, snap_add_thres))
            snapped_labels = list(map(snap_pred,labels,snap_thres, snap_add_thres))
            #f1_batch = f1_score(labels, f1_pred)
            f1_batch = f1_score(snapped_labels, f1_pred)
            prec_batch = precision_score(snapped_labels, f1_pred)
            recall_batch = recall_score(snapped_labels, f1_pred)
            if verbosity >=2:
                print(f"Epoch: {epoch+1:4} {(batch_no+1):8} Batch loss: {c_train_loss:>7f} Batch F1: {f1_batch}")
            train_loss += c_train_loss
            train_f1 += f1_batch
            train_recall += recall_batch
            train_prec += prec_batch
        
        val_loss= 0
        val_f1 = 0
        val_recall= 0
        val_prec = 0
        model.eval()
        for no,(graphs, labels) in enumerate(val_dataloader):
            feats = graphs.ndata['node_features']
            edge_types = graphs.edata['rel_type']
            pred = model(graphs, feats, edge_types)
            c_val_loss = F.mse_loss(pred, labels).item()
            if loss_type == 'cross-entropy':
                c_val_loss = F.cross_entropy(pred, labels).item()
            elif loss_type=='mse':
                c_val_loss = F.mse_loss(pred, labels).item()
            #test_loss += loss_fn(pred, torch.reshape(sample['target'],(1,1))).item()
            val_loss += c_val_loss
            snap_thres = [pred_thres for x in pred]
            snap_add_thres = [add_thres for x in pred]
            f1_pred_val = list(map(snap_pred,pred,snap_thres, snap_add_thres))
            snapped_lebls = list(map(snap_pred,labels, snap_thres, snap_add_thres))
            
            #f1_batch_val = f1_score(labels, f1_pred_val)
            f1_batch_val = f1_score(snapped_lebls, f1_pred_val)
            val_f1 += f1_batch_val
            
            prec_batch_val = precision_score(snapped_lebls, f1_pred_val)
            val_prec += prec_batch_val
            
            recall_batch_val = recall_score(snapped_lebls, f1_pred_val)
            val_recall += recall_batch_val
        test_loss= 0
        test_f1 = 0
        test_recall= 0
        test_prec = 0
        model.eval()
        for no,(graphs, labels) in enumerate(test_dataloader):
            feats = graphs.ndata['node_features']
            edge_types = graphs.edata['rel_type']
            pred = model(graphs, feats, edge_types)
            #c_test_loss = F.mse_loss(pred, labels).item()
            if loss_type == 'cross-entropy':
                c_test_loss = F.cross_entropy(pred, labels).item()
            elif loss_type=='mse':
                c_test_loss = F.mse_loss(pred, labels).item()
            #test_loss += loss_fn(pred, torch.reshape(sample['target'],(1,1))).item()
            test_loss += c_test_loss
            snap_thres = [pred_thres for x in pred]
            snap_add_thres = [add_thres for x in pred]
            f1_pred_test = list(map(snap_pred,pred, snap_thres, snap_add_thres))
            snapped_test = list(map(snap_pred,labels, snap_thres, snap_add_thres))
            
            #f1_batch_val = f1_score(labels, f1_pred_val)
            f1_batch_test = f1_score(snapped_test, f1_pred_test)
            test_f1 += f1_batch_test
            prec_batch_test = precision_score(snapped_test, f1_pred_test)
            test_prec += prec_batch_test
            recall_batch_test = recall_score(snapped_test, f1_pred_test)
            test_recall += recall_batch_test
        
        test_loss= test_loss/len(test_dataloader)
        metric_data['test_loss'].append(test_loss)
        test_f1= test_f1/len(test_dataloader)
        metric_data['test_f1'].append(test_f1)
        test_prec= test_prec/len(test_dataloader)
        metric_data['test_prec'].append(test_prec)
        test_recall= test_recall/len(test_dataloader)
        metric_data['test_recall'].append(test_recall)
        
        val_loss= val_loss/len(val_dataloader)
        metric_data['val_loss'].append(val_loss)
        val_f1= val_f1/len(val_dataloader)
        metric_data['val_f1'].append(val_f1)
        val_prec= val_prec/len(val_dataloader)
        metric_data['val_prec'].append(val_prec)
        val_recall= val_recall/len(val_dataloader)
        metric_data['val_recall'].append(val_recall)
        
        train_loss = train_loss/len(train_dataloader)
        metric_data['train_loss'].append(train_loss)
        train_f1 = train_f1/len(train_dataloader)
        metric_data['train_f1'].append(train_f1)
        train_recall = train_recall/len(train_dataloader)
        metric_data['train_recall'].append(train_recall)
        train_prec = train_prec/len(train_dataloader)
        metric_data['train_prec'].append(train_prec)        
        
        train_f1_hist.append(train_f1)
        val_f1_hist.append(val_f1)
        
        p_val_hist = val_hist
        val_hist.append(val_loss)
        if path_to_save != None and (prev_val_loss == None or val_loss < prev_val_loss):
            torch.save(model,f"{path_to_save}/model_{epoch+1}.pt")
            prev_val_loss = val_loss
            best_epoch = epoch+1
        if path_to_save != None and (val_f1 > best_f1):
            torch.save(model,f"{path_to_save}/best_f1_model_{epoch+1}.pt")
            best_model_path = f"{path_to_save}/best_f1_model_{epoch+1}.pt"
            best_f1 = val_f1
        if verbosity >=2:
            print(f"Train Avg Loss {epoch+1:4}: {train_loss:>8f}\n")
            print(f"Train Avg F1 {epoch+1:4}: {train_f1}\n")
            print(f"Val Avg Loss {epoch+1:4}: {val_loss:>8f}\n")
            print(f"Val Avg F1 {epoch+1:4}:  {val_f1}\n")
            print(f"Optimal Val loss (Epoch {best_epoch}): {prev_val_loss}\n")
        
        if (early_stop < len(p_val_hist)) and (np.sum([1 for v_l in p_val_hist[-early_stop:] if v_l <= val_loss]) == early_stop):
            if verbosity >=1:
                print(f"Early Stopping invoked after epoch {epoch+1}")
                json.dump(metric_data, open(result_path,'w'))
                print("Done!")
            return
    print("Done!")
    json.dump(metric_data, open(result_path,'w'))
    return best_model_path
    
def predict_dgl_graph(model, graph):
    feats = graph.ndata['node_features']
    edge_types = graph.edata['rel_type']
    pred = model(graph, feats, edge_types)
    return pred

def predictBgp2samples(bgps,path, featurizer, model, leaf=False, model_thres=0.5, add_thres = None):
    train_dct = {}
    for x in bgps:
        dgl_graph = tps_graph_const(x, featurizer=featurizer)
        if dgl_graph != None:
            #dgl_graph = tps_to_dgl(tps_graph_const(x, featurizer=featurizer))
            start = time.time()
            p = predict_dgl_graph(model, tps_to_dgl(dgl_graph))
            inference_time = time.time()-start
            p = snap_pred(p,  model_thres, add_thres)
            x.data_dict['prediction'] = p if isinstance(p,int) else p.item()
            x.data_dict['inference_time'] = inference_time
            x.data_dict['bloom_runtime'] =  x.data_dict['bloom_runtime']*1e-9
            if leaf:
                x.data_dict['leapfrog'] =  x.data_dict['leapfrog']*1e-9
            x.data_dict['jena_runtime'] =  x.data_dict['jena_runtime']*1e-9
        train_dct[x.bgp_string] = x.data_dict
    json.dump(train_dct, open(f"{path[:-5]}_w_pred.json",'w'))
    return train_dct

def predict2samples(model,train_path, val_path, test_path, community_no=10, 
                    verbose=True, clust_verbose=False, 
                    clust_load_path='/work/data/confs/May2/pred_clust.json',
                    model_thres=0.5, add_thres = None):
    featurizer = BaseFeaturizer( train_log = train_path, is_pred_clust_feat=True,save_pred_graph_png = None, community_no=community_no,path_pred_clust = { 'save_path':None, 'load_path':clust_load_path}, verbose=clust_verbose)
    
    train_bgps = load_BGPS_from_json(train_path)
    train_data = predictBgp2samples(train_bgps, train_path, featurizer, model,model_thres=model_thres, add_thres = add_thres)
    
    val_bgps = load_BGPS_from_json(val_path)
    val_data = predictBgp2samples(val_bgps, val_path, featurizer, model,model_thres=model_thres, add_thres = add_thres)
    
    test_bgps = load_BGPS_from_json(test_path)
    test_data = predictBgp2samples(test_bgps, test_path, featurizer, model,model_thres=model_thres, add_thres = add_thres)
    return train_data, val_data,test_data

if __name__ == "__main__":
    
    args = parse_arguments()
    
    train_file = args.train_file
    val_file = args.val_file
    test_file = args.test_file
    
    
    path_to_save = '/work/data/models'
    train_file = '/work/data/splits/splits_0.05/train.json'
    val_file = '/work/data/splits/splits_0.05/val.json'
    test_file = '/work/data/splits/splits_0.05/test.json'
    os.system('mkdir -p /work/data/models2')
    path_to_save = '/work/data/models2'
    path_to_res = '/work/data/results/temp.txt'
    model = Classifier(113, 20, 2)
    predict2samples(model,train_file, val_file, test_file, community_no=10, verbose=True, clust_verbose=False, clust_load_path='/work/data/confs/May2/pred_clust.json')
    exit()
    train_dataloader, val_dataloader, test_dataloader = extract_data(train_file, val_file, test_file, community_no=30, batch_size = 50, verbose=True)