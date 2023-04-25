from classifier.bgp_dataset import BGPDataset
from feature_extraction.predicate_features_sub_obj import Predicate_Featurizer_Sub_Obj
from feature_extraction.predicate_features import PredicateFeaturesQuery
from feature_extraction.entity_features import EntityFeatures
import torch
from graph_construction import triple_pattern
from graph_construction.bgp_graph import BGPGraph
from graph_construction.bgp import BGP
from graph_construction.nodes.node import Node
from utils import bgp_graph_construction, load_BGPS_from_json

import configparser
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, HeteroData
from feature_extraction.constants import PATH_TO_CONFIG_GRAPH

def get_graph_representation(parser, data_file, node=Node) -> None:
    #super().__init__(parser, transform, pre_transform, pre_filter)
    feat_generation_path = parser['PredicateFeaturizerSubObj']['load_path']
    topk = int(parser['PredicateFeaturizerSubObj']['topk'])
    bin_no = int(parser['PredicateFeaturizerSubObj']['bin_no'])
    pred_feature_rizer = Predicate_Featurizer_Sub_Obj.prepare_pred_featues_for_bgp(feat_generation_path, bins=bin_no, topk=topk)
    
    #node.pred_feaurizer = pred_feature_rizer
    #bgps = load_BGPS_from_json(data_file, node=node)
    node.pred_feaurizer = pred_feature_rizer
    node.ent_featurizer = None
    node.pred_bins = bin_no
    node.pred_topk = topk
    node.pred_feat_sub_obj_no = True
    node.use_ent_feat = False
    bgps = load_BGPS_from_json(data_file, node=node)
    total_bgps = len(bgps)
    
    bgp_graphs = bgp_graph_construction(bgps, return_graphs=True, filter=True)
    #bgp_graphs = bgp_graphs[:5]
    #print(f"Removed {total_bgps-len(bgp_graphs)} of {total_bgps}")
    
    node_features = []
    edge_lsts = []
    target = []
    join_indices:list[int] = []
    
    new_bgp_graph:list[BGPGraph] = []
    hetero_data = []
    for inum,g in enumerate(bgp_graphs):
        g:BGPGraph
        #Nan check
        #node_feat = torch.tensor( g.get_node_representation(bin_no,topk, pred_feat_sub_obj_no=True, use_ent_feat=True, ent_bins = entity_featurizer.buckets), dtype=torch.float32)
        node_feat = torch.tensor( g.get_node_representation(), dtype=torch.float32)
        #if torch.sum(torch.isnan( node_feat)) > 0:
        #    continue
        new_bgp_graph.append(g)
        target.append(g.gt)
        join_indices.append(g.last_join_index)
        edge_g = torch.tensor(g.get_edge_list(), dtype=torch.int64)
        edge_lsts.append(  edge_g)
        h_data = Data(x=node_feat,edge_index=edge_g, y= torch.tensor([g.gt], dtype=torch.float32).reshape((1,1)) )
        h_data.id = inum
        #h_data['node_feat'] = node_feat
        #h_data['node_feat'].edge_index = edge_g
        hetero_data.append(h_data)
        #node_feat = torch.nan_to_num(node_feat,-1)
        node_features.append( node_feat)
        
    
    #target = target
    bgp_graphs = new_bgp_graph
    
    print(f"Removed {total_bgps-len(bgp_graphs)} of {total_bgps}")
    return hetero_data

def get_graph_for_single_sample(bgp_string=None, ground_truth=None,  bgp_graph = None, node=Node):
    if bgp_graph == None:
        bgp = BGP(bgp_string, ground_truth,node_class=node)
        bgp_graph = BGPGraph(bgp)
        node_feat = torch.tensor( bgp_graph.get_node_representation(), dtype=torch.float32)
        edge_g = torch.tensor(bgp_graph.get_edge_list(), dtype=torch.int64)
        h_data = Data(x=node_feat,edge_index=edge_g, y= torch.tensor([bgp_graph.gt], dtype=torch.float32).reshape((1,1)) )
    else:
        node_feat = torch.tensor( bgp_graph.get_node_representation(), dtype=torch.float32)
        edge_g = torch.tensor(bgp_graph.get_edge_list(), dtype=torch.int64)
        h_data = Data(x=node_feat,edge_index=edge_g, y= torch.tensor([bgp_graph.gt], dtype=torch.float32).reshape((1,1)) )
    #h_data = DataLoader(h_data, batch_size=1)
    return h_data
    
    
if __name__ == "__main__":
    parser = configparser.ConfigParser()
    parser.read(PATH_TO_CONFIG_GRAPH)
    data_file = parser['DebugDataset']['train_data']
    dataset = get_graph_representation(parser,data_file)
    tes= DataLoader(dataset, batch_size=1, shuffle=True)
    ##print(tes.dataset[0].x.shape)
    #print(len(tes))
    for batch in tes:
        #print(f"len: {len(batch)}")
        #print(batch.edge_index)
        exit()
