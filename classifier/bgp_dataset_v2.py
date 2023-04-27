from classifier.bgp_dataset import BGPDataset
from feature_extraction.predicate_features_sub_obj import Predicate_Featurizer_Sub_Obj
from feature_extraction.predicates.predicate_features import PredicateFeaturesQuery
from feature_extraction.entity_features import EntityFeatures
import torch
from graph_construction.bgp_graph import BGPGraph
from graph_construction.bgp import BGP
from graph_construction.nodes.node import Node
from utils import unpickle_obj,pickle_obj, bgp_graph_construction, load_BGPS_from_json,load_obj_w_function
from feature_extraction.constants import PATH_TO_CONFIG
import configparser
from torch_geometric.data import Dataset

class BGPDataset_v2:#(Dataset):
    def __init__(self, parser, data_file, pickle_file=None, transform = None, target_transform = None, node = Node) -> None:
        #super().__init__(parser, transform, pre_transform, pre_filter)
        feat_generation_path = parser['PredicateFeaturizerSubObj']['load_path']
        topk = int(parser['PredicateFeaturizerSubObj']['topk'])
        bin_no = int(parser['PredicateFeaturizerSubObj']['bin_no'])
        pred_feature_rizer = Predicate_Featurizer_Sub_Obj.prepare_pred_featues_for_bgp(feat_generation_path, bins=bin_no, topk=topk)
        #TODO outcommented for now
        #entity_featurizer = EntityFeatures.load(parser)
        #entity_featurizer.create_bins(int(parser['EntityFeaturizer']['bin_no']))
        
        #bgps = load_obj_w_function(data_file,pickle_file,load_BGPS_from_json, pred_feat=pred_feature_rizer, ent_feat=entity_featurizer)
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
        
        self.node_features = []
        self.edge_lsts = []
        self.target = []
        self.join_indices:list[int] = []
        
        new_bgp_graph:list[BGPGraph] = []
        for g in bgp_graphs:
            g:BGPGraph
            #Nan check
            #node_feat = torch.tensor( g.get_node_representation(bin_no,topk, pred_feat_sub_obj_no=True, use_ent_feat=True, ent_bins = entity_featurizer.buckets), dtype=torch.float32)
            node_feat = torch.tensor( g.get_node_representation(), dtype=torch.float32)
            #if torch.sum(torch.isnan( node_feat)) > 0:
            #    continue
            new_bgp_graph.append(g)
            self.target.append(g.gt)
            self.join_indices.append(g.last_join_index)
            
            
            #node_feat = torch.nan_to_num(node_feat,-1)
            self.node_features.append( node_feat)
            self.edge_lsts.append( torch.tensor(g.get_edge_list(), dtype=torch.int64) )
        
        #self.target = self.target
        self.bgp_graphs = new_bgp_graph
        
        print(f"Removed {total_bgps-len(self.bgp_graphs)} of {total_bgps}")
        
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, index) -> dict:
        node_features = self.node_features[index]
        edges = self.edge_lsts[index]
        target = torch.tensor( [self.target[index]], dtype=torch.float32)
        join_index = self.join_indices[index]
        if self.transform != None:
            node_features = self.transform(node_features)
        if self.target_transform != None:
            target = self.target_transform(target)
        sample = {'nodes':node_features,'edges':edges,'target':target, 'join_index':join_index}
        return sample#(node_features,edges,target)
if __name__ == "__main__":
    parser = configparser.ConfigParser()
    parser.read(PATH_TO_CONFIG)
    
    data_file = parser['Dataset']['train_data_path']
    pickle_file = parser['Dataset']['train_data_bgp_pickle']
    dataset = BGPDataset_v2(parser,data_file, pickle_file)
    print(dataset[0])