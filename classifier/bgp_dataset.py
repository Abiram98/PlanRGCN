from torch.utils.data import Dataset
from utils import unpickle_obj,pickle_obj, filter_bgps_w_missing_pred_feat, load_BGPS_from_json,load_obj_w_function
from feature_extraction.predicate_features import PredicateFeaturesQuery
from graph_construction.bgp_graph import BGPGraph
import torch, math


class BGPDataset(Dataset):
    def __init__(self, data_file,feat_generation_path, pickle_file,bin=30, transform=None, target_transform=None) -> None:
        pred_feature_rizer = PredicateFeaturesQuery.prepare_pred_featues_for_bgp( feat_generation_path, bins=bin)
        bgps = load_obj_w_function(data_file,pickle_file,load_BGPS_from_json, pred_feat=pred_feature_rizer)
        #bgps  = unpickle_obj(pickle_file)
        #if bgps == None:
        #    bgps = load_BGPS_from_json(data_file, pred_feat=pred_feature_rizer)
        #    pickle_obj(bgps, data_file)
        total_bgps = len(bgps)
        bgp_graphs = filter_bgps_w_missing_pred_feat(bgps, return_graphs=True)
        print(f"Filtered {total_bgps-len(bgp_graphs)} of {total_bgps}")
        
        self.node_features = []
        self.edge_lsts = []
        self.target = []
        self.join_indices:list[int] = []
        for bgp in bgp_graphs:
            bgp:BGPGraph
            self.target.append(bgp.gt)
            self.join_indices.append(bgp.last_join_index)
            node_feat = torch.FloatTensor( bgp.get_node_representation(bin))
            #node_feat = torch.nan_to_num(node_feat,-1)
            self.node_features.append( node_feat)
            self.edge_lsts.append( torch.IntTensor(bgp.get_edge_list()) )
        
        self.target = torch.FloatTensor([self.target])
        
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, index) -> tuple:
        node_features = self.node_features[index]
        edges = self.edge_lsts[index]
        target = self.target[index]
        join_index = self.join_indices[index]
        if self.transform != None:
            node_features = self.transform(node_features)
        if self.target_transform != None:
            target = self.target_transform(target)
        sample = {'nodes':node_features,'edges':edges,'target':target, 'join_index':join_index}
        return sample#(node_features,edges,target)

from torch.utils.data import DataLoader
if __name__ == "__main__":
    bins = 30
    el_in_batch= 200
    data_file = '/work/data/train_data.json'
    feat_generation_path = '/work/data/pred_feat.pickle'
    pickle_file = '/work/data/bgps.pickle'
    dataset = BGPDataset(data_file,feat_generation_path, pickle_file,bin=bins)
    #print(dataset[0])
    loader = DataLoader(dataset, batch_size=math.ceil(len(dataset)/el_in_batch), shuffle=True)
    pickle_obj(loader,)
    node_features,edges,target = next(iter(loader))
    print(len(target))
   
#path_to_bgps = '/work/data/bgps.pickle'
#path_predicate_feat_gen = '/work/data/pred_feat.pickle'
#bgps = unpickle_obj(path_to_bgps)
#bins = 30
#limit_BGPs = None
#if bgps == None:
#    bgps = load_BGPS_from_json('/work/data/train_data.json', path_predicate_feat_gen=path_predicate_feat_gen,limit_bgp=limit_BGPs, bins=bins)
#    pickle_obj(bgps,path_to_bgps)
#bgp_count = len(bgps)
#pred_featurizer = PredicateFeaturesQuery.prep#are_pred_featues_for_bgp(path_predicate_feat_gen, bins=bins)
#bgps = filter_bgps_w_missing_pred_feat(bgps, pred_featurizer)
#print(f"Filtered {bgp_count-len(bgps)} of {bgp_count}")   