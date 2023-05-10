import math,os
import torch.nn as nn, torch, numpy as np
from torch_geometric import seed_everything
from classifier.batched.gcn import GNN, GNN_w_Dense
import configparser

from feature_extraction.predicate_features_sub_obj import Predicate_Featurizer_Sub_Obj
from classifier.batched.hetero_dataset import get_graph_representation
from feature_extraction.constants import PATH_TO_CONFIG_GRAPH
from torch_geometric.loader import DataLoader
from feature_extraction.predicates.pred_feat_num_enc import Pred_Feat_Num_Enc
from feature_extraction.predicates.predicate_clust_feat import Pred_clust_feat
from feature_extraction.predicates.ql_pred_featurizer import ql_pred_featurizer
from graph_construction.bgp_graph import BGPGraph
from graph_construction.nodes.cluster_node import Cluster_node

from graph_construction.nodes.node import Node
from graph_construction.nodes.node_num_pred import Node_num_pred_encoding
from graph_construction.nodes.ql_node import ql_node
from preprocessing.utils import get_predicates_from_path

#NODE = ql_node
#PRED_FEATURIZER = ql_pred_featurizer
NODE = Node
PRED_FEATURIZER = Predicate_Featurizer_Sub_Obj
#NODE = Node_num_pred_encoding
#PRED_FEATURIZER = Pred_Feat_Num_Enc
#NODE = Cluster_node
#PRED_FEATURIZER = Pred_clust_feat


torch.manual_seed(12345)
np.random.seed(12345)
seed_everything(12345)
from sklearn.metrics import f1_score,precision_score,recall_score, accuracy_score
def run_undivided(train_loader:DataLoader, validation_loader:DataLoader, path_to_save ='/work/data/confs/newPredExtractionRun/', epoch=50, early_stop=10, layer_size=100):
    def snap_pred(pred):
        if pred >= 0.5:
            return 1
        else:
            return 0
    
    #global train_loader
    #train_loader = return_graph_dataloader(dataset, batch_size=BATCH_SIZE)
    model = GNN(train_loader.dataset[0].x.shape[1] , train_loader.dataset[0].x.shape[1]*2)
    model = model.float()

    #optimizer and loss function
    print("Loss function: MSELoss")
    loss_fn = nn.MSELoss()
    #print("Loss function: BCELoss")
    #loss_fn = nn.BCELoss()
    print(f"Optimizer: LR [{LR}] weight decay [{WEIGHT_DECAY}]")
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    
    prev_val_loss = None
    val_hist = []
    best_epoch = 0
    for t in range(epoch):
        print(f"Epoch {t+1}\n--------------------------------------------------------------")

        train_loss = 0
        train_f1 = 0
        model.train()
        for sample_no, batch in enumerate(train_loader):
            pred = model(batch)
            #optimizer.zero_grad()
            #loss = loss_fn(pred, torch.reshape(sample['target'],(1,1)))
            loss = loss_fn(pred, batch.y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            c_train_loss = loss.item()
            f1_pred = list(map(snap_pred,pred))
            f1_batch = f1_score(batch.y, f1_pred)
            print(f"Epoch: {t+1:4} {(sample_no+1):8} Batch loss: {c_train_loss:>7f} Batch F1: {f1_batch}")
            train_loss += c_train_loss
            train_f1 += f1_batch
        val_loss= 0
        val_f1 = 0
        with torch.no_grad():
            for no,sample in enumerate(validation_loader):
                pred = model(sample)
                c_test_loss = loss_fn(pred, sample.y).item()
                #test_loss += loss_fn(pred, torch.reshape(sample['target'],(1,1))).item()
                val_loss += c_test_loss
                f1_pred_val = list(map(snap_pred,pred))
                f1_batch_val = f1_score(sample.y, f1_pred_val)
                val_f1 += f1_batch_val
        val_loss= val_loss/len(val_loader)
        val_f1= val_f1/len(val_loader)
        train_loss = train_loss/len(train_loader)
        train_f1 = train_f1/len(train_loader)
        
        p_val_hist = val_hist
        val_hist.append(val_loss)
        if path_to_save != None and (prev_val_loss == None or val_loss < prev_val_loss):
            torch.save(model,f"{path_to_save}model_{t+1}.pt")
            prev_val_loss = val_loss
            best_epoch = t+1
        
        print(f"Train Avg Loss {t+1:4}: {train_loss:>8f}\n")
        print(f"Train Avg F1 {t+1:4}: {train_f1}\n")
        print(f"Val Avg Loss {t+1:4}: {val_loss:>8f}\n")
        print(f"Val Avg F1 {t+1:4}:  {val_f1}\n")
        print(f"Optimal Val loss (Epoch {best_epoch}): {prev_val_loss}\n")
        
        if (early_stop > len(p_val_hist)) and (np.sum([1 for v_l in p_val_hist[-early_stop:] if v_l <= val_loss]) == early_stop):
            print(f"Early Stopping invoked after epoch {t+1}")
            exit()
    print("Done!")

def get_debug_data_loader(parser):
    BATCH_SIZE = int(parser['Training']['BATCH_SIZE'])
    data_file = parser['DebugDataset']['train_data']
    val_file = parser['DebugDataset']['val_data']
    test_file = parser['DebugDataset']['test_data']
    feat_generation_path = parser['PredicateFeaturizerSubObj']['load_path']
    train_bgp_file_pickle = parser['DebugDataset']['train_bgp_file_pickle']
    val_bgp_file_pickle = parser['DebugDataset']['val_bgp_file_pickle']
    test_bgp_file_pickle = parser['DebugDataset']['test_bgp_file_pickle']
    
    pickle_train_dataset = parser['DebugDataset']['pickle_train_dataset']
    pickle_testdataset = parser['DebugDataset']['pickle_testdataset']
    pickle_valdataset = parser['DebugDataset']['pickle_valdataset']
    
    feat_generation_path = parser['PredicateFeaturizerSubObj']['load_path']
    topk = int(parser['PredicateFeaturizerSubObj']['topk'])
    bin_no = int(parser['PredicateFeaturizerSubObj']['bin_no'])
    pred_feature_rizer = PRED_FEATURIZER.prepare_pred_featues_for_bgp(feat_generation_path, bins=bin_no, topk=topk, obj_type=PRED_FEATURIZER)
    if isinstance(pred_feature_rizer, ql_pred_featurizer):
        preds = get_predicates_from_path(data_file)
        pred_feature_rizer.prepare_featurizer(preds,topk,bin_no)
    elif isinstance(pred_feature_rizer,Pred_clust_feat):
        pred_feature_rizer.create_cluster_from_pred_file(data_file, save_pred_graph_png=parser['PredicateCluster']['pred_graph'])
        NODE.max_pred_buckets = pred_feature_rizer.max_clusters
    #node.pred_feaurizer = pred_feature_rizer
    #bgps = load_BGPS_from_json(data_file, node=node)
    BGPGraph.node_type = NODE
    NODE.pred_feaurizer = pred_feature_rizer
    NODE.ent_featurizer = None
    NODE.pred_bins = bin_no
    NODE.pred_topk = topk
    NODE.pred_feat_sub_obj_no = True
    NODE.use_ent_feat = False
    
    train_loader = DataLoader(get_graph_representation(data_file, node=NODE), batch_size=BATCH_SIZE, shuffle=True)
    #train_loader = DataLoader(get_graph_representation(data_file, node=NODE,norm=None), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(get_graph_representation(val_file, node=NODE), batch_size=BATCH_SIZE, shuffle=True)
    #val_loader = DataLoader(get_graph_representation(val_file, node=NODE, norm=None), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(get_graph_representation(test_file, node=NODE), batch_size=BATCH_SIZE, shuffle=True)
    #test_loader = DataLoader(get_graph_representation(test_file, node=NODE, norm=None), batch_size=BATCH_SIZE, shuffle=True)
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    
    parser = configparser.ConfigParser()
    parser.read(PATH_TO_CONFIG_GRAPH)
    
    #Model hyper parameters
    EPOCHS = int(parser['Training']['EPOCHS'])
    BATCH_SIZE = int(parser['Training']['BATCH_SIZE'])
    LR = float(parser['Training']['LR'])
    WEIGHT_DECAY = float(parser['Training']['WEIGHT_DECAY'])
    PREDICATE_BINS = int(parser['Training']['PREDICATE_BINS'])
    
    train_loader, val_loader, test_loader = get_debug_data_loader(parser)
    #for batch in train_loader:
    #    print(batch.y)
    #exit()
    #torch.save(train_loader,pickle_train_dataset)
    #torch.save(test_loader,pickle_testdataset)
    #torch.save(val_loader,pickle_valdataset)
        
    #
    #temporary code
    #def nan_yielder(dataset):
    #    for sample in dataset:
    #        if torch.sum(torch.isnan( sample['nodes'])) > 0:
    #            yield sample
    #gen = nan_yielder(dataset)

    #Model definition
    
    
    #run(dataset, test_dataset,loss_fn,optimizer, epoch=EPOCHS, path_to_save=parser['Results']['path_to_save_model'])
    print("Model with variable positions in join nodes")
    run_undivided(train_loader, val_loader, epoch=EPOCHS, path_to_save=parser['Results']['path_to_save_model_batched'], layer_size=int(parser['Training']['layer_size']))