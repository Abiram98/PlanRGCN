from torch.utils.data import Dataset
from graph_construction.nodes.node import Node
from preprocessing.utils import unpickle_obj,pickle_obj, bgp_graph_construction, load_BGPS_from_json,load_obj_w_function
from feature_extraction.predicates.predicate_features import PredicateFeaturesQuery
from graph_construction.bgp_graph import BGPGraph
import torch, math
from torch_geometric.data import HeteroData, Data
from torch_geometric.loader import HGTLoader, DataLoader

class BGPDataset(Dataset):
    def __init__(self, data_file,feat_generation_path, pickle_file,bin=30, topk=15, transform=None, target_transform=None, pred_feat_sub_obj_no=False, node=Node) -> None:
        pred_feature_rizer = PredicateFeaturesQuery.prepare_pred_featues_for_bgp( feat_generation_path, bins=bin, topk=topk)
        #bgps = load_obj_w_function(data_file,pickle_file,load_BGPS_from_json, pred_feat=pred_feature_rizer)
        node.pred_feaurizer = pred_feature_rizer
        node.ent_featurizer = None
        node.pred_bins = bin
        node.pred_topk = topk
        node.pred_feat_sub_obj_no = True
        node.use_ent_feat = False
        bgps = load_BGPS_from_json(data_file, node=node)
        #bgps  = unpickle_obj(pickle_file)
        #if bgps == None:
        #    bgps = load_BGPS_from_json(data_file, pred_feat=pred_feature_rizer)
        #    pickle_obj(bgps, data_file)
        
        total_bgps = len(bgps)
        #TODO outcomment this
        
        bgp_graphs = bgp_graph_construction(bgps, return_graphs=True, filter=True)
        #bgp_graphs = bgp_graphs[:5]
        #print(f"Removed {total_bgps-len(bgp_graphs)} of {total_bgps}")
        
        self.node_features = []
        self.edge_lsts = []
        self.target = []
        self.join_indices:list[int] = []
        
        new_bgp_graph:list[BGPGraph] = []
        for bgp in bgp_graphs:
            bgp:BGPGraph
            #Nan check
            node_feat = torch.tensor( bgp.get_node_representation(), dtype=torch.float32)
            if torch.sum(torch.isnan( node_feat)) > 0:
                continue
            new_bgp_graph.append(bgp)
            self.target.append(bgp.gt)
            self.join_indices.append(bgp.last_join_index)
            
            
            #node_feat = torch.nan_to_num(node_feat,-1)
            self.node_features.append( node_feat)
            self.edge_lsts.append( torch.tensor(bgp.get_edge_list(), dtype=torch.int64) )
        
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
    
#This will not work    
def return_graph_dataloader(dataset:BGPDataset, batch_size, shuffle=True) -> DataLoader:
        lst_data:list[Data] = []
        for i in range(len(dataset)):
            sample = dataset[i]
            node_feat, edges, target, join_index = sample['nodes'], sample['edges'], sample['target'], sample['join_index']
            d1 = Data(x=node_feat,edge_index=edges, y=target)
            d1.join_index = join_index
            
            lst_data.append(d1)
            
        return DataLoader(lst_data,batch_size=batch_size,shuffle=shuffle)

#from torch.utils.data import DataLoader
"""if __name__ == "__main__":
    bins = 30
    el_in_batch= 200
    topk = 15
    data_file = '/work/data/train_data.json'
    feat_generation_path = '/work/data/confs/newPredExtractionRun/pred_feat_01_04_2023_07_48.pickle'
    pickle_file = '/work/data/bgps.pickle'
    dataset = BGPDataset(data_file,feat_generation_path, pickle_file,bin=bins, topk=topk)
    sample = dataset[0]
    print(sample)"""
    #print(dataset[0])
    #loader = DataLoader(dataset, batch_size=math.ceil(len(dataset)/el_in_batch), shuffle=True)
    #pickle_obj(loader,)
    #node_features,edges,target = next(iter(loader))
    #print(len(target))
   
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