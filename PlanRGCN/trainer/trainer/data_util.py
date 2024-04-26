from graph_construction.feats.featurizer import FeaturizerBase, FeaturizerPredStats
from graph_construction.feats.featurizer_pathv2 import FeaturizerPathV2
from graph_construction.query_graph import (
    QueryPlan,
    QueryPlanCommonBi,
    query_graph_w_class_vec,
    snap_lat2onehot,
    snap_lat2onehotv2,
)

from dgl.dataloading import GraphDataLoader
from dgl import save_graphs, load_graphs
from dgl.data.utils import makedirs, save_info, load_info
import os
import torch as th
from pathlib import Path
class GraphDataset:
    def __init__(self, graphs, labels, ids, save_path, vec_size, scaling, literals=False,durationQPS=None, act_save=None) -> None:
        self.graph =  graphs
        self.labels = labels
        self.ids = ids
        self.durationQPS = durationQPS
        self.save_path = save_path
        self.vec_size = vec_size
        self.scaling = scaling
        self.featurizer:FeaturizerBase = None
        self.query_plan :QueryPlan= None
        self.is_literals = literals
        self.act_save = act_save

    def __getitem__(self, i):
        return self.graph[i], self.labels[i], self.ids[i]

    def __len__(self):
        return len(self.labels)

    def get_paths(self):
        if self.act_save != None:
            dir_path = self.act_save
        elif self.is_literals:
            dir_path = os.path.join(Path(self.save_path).parent.absolute(),f"planrgcn_{self.scaling}_litplan")
        else:
            dir_path = os.path.join(Path(self.save_path).parent.absolute(),f"planrgcn_{self.scaling}")
        
        file_name = Path(self.save_path).name
        if file_name.endswith(".tsv"):
            file_name = file_name.replace(".tsv", "")
        graph_path = os.path.join(dir_path , f'{file_name}_dgl_graph.bin')
        info_path = os.path.join(dir_path , f'{file_name}_info.pkl')
        return graph_path, info_path
    
    def save(self):
        # save graphs and labels
        graph_path, info_path= self.get_paths()
        print(f"GraphDataset saved at {graph_path}")
        save_graphs(graph_path, self.graph, {'labels': self.labels})
        # save other information in python dict
        save_info(info_path, {'ids':self.ids, 'vec_size':self.vec_size, "featurizer":self.featurizer, "query_plan": self.query_plan})

    def load(self):
        # load processed data from directory `self.save_path`
        # save graphs and labels
        graph_path, info_path= self.get_paths()
        #graph_path = self.save_path +'_dgl_graph.bin'
        self.graph, label_dict = load_graphs(graph_path)
        self.labels = label_dict['labels']
        #info_path = self.save_path+ '_info.pkl'
        info_dict = load_info(info_path)
        self.ids = info_dict['ids']
        self.vec_size = info_dict['vec_size']
        self.featurizer = info_dict['featurizer']
        self.query_plan = info_dict['query_plan']

    def load_dataset(path, scaling, lp, act_save=None):
        if lp is not None:
            path += "_litplan"
            temp =GraphDataset([],[],[], path, 0, scaling,  literals=True, act_save=act_save)
        else:
            temp =GraphDataset([],[],[], path, 0, scaling, act_save=act_save)
        if temp.has_cache():
            temp.load()
            return temp
        return None
    
    def has_cache(self):
        # check whether there are processed data in `self.save_path`
        graph_path, info_path= self.get_paths()
        return os.path.exists(graph_path) and os.path.exists(info_path)
    
    def set_query_plan(self, query_plan):
        self.query_plan = query_plan
    
    def set_featurizer(self,feat):
        self.featurizer = feat


class DatasetPrep:
    def __init__(
        self,
        train_path="/qpp/dataset/DBpedia_2016_12k_sample/train_sampled.tsv",
        val_path="/qpp/dataset/DBpedia_2016_12k_sample/val_sampled.tsv",
        test_path="/qpp/dataset/DBpedia_2016_12k_sample/test_sampled.tsv",
        val_pp_path = None,
        batch_size=64,
        query_plan_dir="/PlanRGCN/extracted_features/queryplans/",
        pred_stat_path="/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats",
        pred_com_path="/PlanRGCN/data/pred/pred_co/pred2index_louvain.pickle",
        pred_end_path=None,
        ent_path="/PlanRGCN/extracted_features/entities/ent_stat/batches_response_stats",
        lit_path = None,
        time_col="mean_latency",
        cls_func=snap_lat2onehotv2,
        featurizer_class=FeaturizerPredStats,
        query_plan=QueryPlan,
        is_lsq=False,
        scaling="None",
        debug = False,
        save_path=None
    ) -> None:
        self.train_path = train_path
        self.val_path = val_path
        self.test_path = test_path
        self.cls_func = cls_func
        self.val_pp_path = val_pp_path
        self.save_path = save_path
        
        self.time_col = time_col
        self.feat = featurizer_class(
            pred_stat_path,
            pred_com_path=pred_com_path,
            ent_path=ent_path,
            lit_path=lit_path,
            #pred_end_path=pred_end_path,
            scaling=scaling,
        )
        self.vec_size = self.feat.filter_size + self.feat.tp_size
        if isinstance(self.feat, FeaturizerPathV2):
            self.vec_size += self.feat.pp_size
        self.query_plan_dir = query_plan_dir
        self.batch_size = batch_size
        self.query_plan = query_plan
        self.is_lsq = is_lsq
        self.scaling = scaling
        self.debug = debug

    def get_dataloader(self, path):
        (graphs, clas_list, ids),durationQPS = query_graph_w_class_vec(
            self.query_plan_dir,
            query_path=path,
            feat=self.feat,
            time_col=self.time_col,
            cls_funct=self.cls_func,
            query_plan=self.query_plan,
            is_lsq=self.is_lsq,
            debug = self.debug
        )
        train_dataset = GraphDataset(graphs, clas_list, ids, path, self.vec_size, self.scaling,durationQPS=durationQPS,act_save=self.save_path)
        train_dataset.set_query_plan(self.query_plan)
        train_dataset.set_featurizer(self.feat)
        train_dataloader = GraphDataLoader(
            train_dataset, batch_size=self.batch_size, drop_last=False, shuffle=True
        )
        return train_dataloader

    def get_trainloader(self):
        return self.get_dataloader(self.train_path)

    def get_valloader(self):
        return self.get_dataloader(self.val_path)

    def get_testloader(self):
        return self.get_dataloader(self.test_path)
    
    def get_pp_valloader(self):
        if not self.val_pp_path is None:
            return self.get_dataloader(self.val_pp_path)
    


if __name__ == "__main__":
    pred_stat_path = (
        "/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats"
    )
    query_plan_dir = "/PlanRGCN/extracted_features/queryplans/"

    feat = FeaturizerPredStats(pred_stat_path)
    batch_size = 2
    train_path = "/qpp/dataset/DBpedia_2016_12k_sample/train_sampled.tsv"
    func = lambda x: x
    graphs, clas_list, ids = query_graph_w_class_vec(
        query_plan_dir,
        query_path=train_path,
        feat=feat,
        time_col="mean_latency",
        cls_funct=func,
    )
    train_dataset = GraphDataset(graphs, clas_list, ids)
    train_dataloader = GraphDataLoader(
        train_dataset, batch_size=batch_size, drop_last=False, shuffle=True
    )
    for g, l, i in train_dataloader:
        print(i)
        break
