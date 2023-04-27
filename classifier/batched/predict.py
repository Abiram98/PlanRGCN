from feature_extraction.constants import PATH_TO_CONFIG_GRAPH
from classifier.batched.hetero_dataset import get_graph_for_single_sample
from classifier.batched.trainer import NODE
import torch
import configparser
from classifier.batched.gcn import GNN
from feature_extraction.predicate_features_sub_obj import Predicate_Featurizer_Sub_Obj
from graph_construction.bgp_graph import BGPGraph
from graph_construction.nodes.node import Node
from utils import bgp_graph_construction, load_BGPS_from_json
from sklearn.metrics import f1_score,precision_score,recall_score, accuracy_score

def predict(model, graph_data):
    prediction = None
    with torch.no_grad():
            prediction = model(graph_data)
    return prediction

def get_metrics(bgp_graphs:list[BGPGraph], model, node=Node):
    preds, truths = [], []
    for i in bgp_graphs:
        h_data = get_graph_for_single_sample(bgp_graph=i, node=node)
        preds.append(model(h_data).item())
        truths.append(h_data.y.item())
    def snap_pred(pred):
        if pred > 0.5:
            return 1
        else:
            return 0
    preds = list(map(snap_pred,preds))
    
    f1, precision, recall, accuracy = f1_score(truths,preds,average='binary'),precision_score(truths,preds,average='binary'),recall_score(truths,preds,average='binary'), accuracy_score(truths,preds)
    print(f"\tF1: {f1}, Precision: {precision}, Recall: {recall}, Accuracy: {accuracy}")
    return f1, precision, recall

if __name__ == "__main__":
    parser = configparser.ConfigParser()
    parser.read(PATH_TO_CONFIG_GRAPH)
    path_to_best_mode = parser['Results']['path_to_best_model']
    model = torch.load(path_to_best_mode)
    feat_generation_path = parser['PredicateFeaturizerSubObj']['load_path']
    topk = int(parser['PredicateFeaturizerSubObj']['topk'])
    bin_no = int(parser['PredicateFeaturizerSubObj']['bin_no'])
    pred_feature_rizer = Predicate_Featurizer_Sub_Obj.prepare_pred_featues_for_bgp(feat_generation_path, bins=bin_no, topk=topk)
    BGPGraph.node_type = NODE
    NODE.pred_feaurizer = pred_feature_rizer
    NODE.ent_featurizer = None
    NODE.pred_bins = bin_no
    NODE.pred_topk = topk
    NODE.pred_feat_sub_obj_no = True
    NODE.use_ent_feat = False
    
    train_data_file = parser['DebugDataset']['train_data']
    train_bgps = load_BGPS_from_json(train_data_file)
    train_bgp_graphs = bgp_graph_construction(train_bgps, return_graphs=True, filter=True)
    h_data = get_graph_for_single_sample(bgp_graph=train_bgp_graphs[0], node=NODE)
    print(h_data.x)
    exit()
    print("Train")
    get_metrics(train_bgp_graphs, model, node=Node)
    #h_data = get_graph_for_single_sample(bgp_graph=bgp_graphs[0], bin_no=bin_no, topk=topk)
    #print(model(h_data).item(), h_data.y.item())
    #for batch in h_data:
        #print(model(batch))
    val_data_file = parser['DebugDataset']['val_data']
    val_bgps = load_BGPS_from_json(val_data_file)
    val_bgp_graphs = bgp_graph_construction(val_bgps, return_graphs=True, filter=True)
    print("Val")
    get_metrics(val_bgp_graphs, model, bin_no, topk)
    
    test_data_file = parser['DebugDataset']['test_data']
    test_bgps = load_BGPS_from_json(test_data_file)
    test_bgp_graphs = bgp_graph_construction(test_bgps, return_graphs=True, filter=True)
    print("Test")
    get_metrics(test_bgp_graphs, model, bin_no, topk)
    