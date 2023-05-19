import json

import numpy as np
from graph_construction.bgp import BGP
from graph_construction.triple_pattern import TriplePattern
from preprocessing.utils import get_bgp_predicates_from_path
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network

import itertools

class BaseFeaturizer:
    def __init__(self, train_log = None, is_pred_clust_feat=True,save_pred_graph_png = None, community_no:int=10, pred_clust_verbose= False,
                 path_preds = {'pred_lits':'/work/data/extracted_statistics/updated_nt_pred_unique_lits.json',
                               'pred_ents':'/work/data/extracted_statistics/updated_nt_pred_unique_subj_obj.json',
                               'pred_freq':'/work/data/extracted_statistics/updated_pred_freq.json'},
                 path_pred_clust = { 'load_path':None, 'save_path':None},
                 verbose=True) -> None:
        self.verbose = verbose
        self.pred_freq = json.load(open(path_preds['pred_freq'])) #contains freqency of predicates
        self.pred_ents = json.load(open(path_preds['pred_ents'])) #contians # of unique subject and object for predicates
        self.pred_lits = json.load(open(path_preds['pred_lits'])) #contains # of unique literal values for predicates
        
        if is_pred_clust_feat:
            self.pred2index = None
            self.max_clusters = None
            self.create_cluster_from_pred_file(train_log,save_pred_graph_png = save_pred_graph_png, 
                                               community_no=community_no, verbose=pred_clust_verbose, 
                                               load_path=path_pred_clust['load_path'], save_path=path_pred_clust['save_path'])
        pass

    def get_feat_vec(self,tp:TriplePattern):
        #tp features
        var_pos = np.zeros(3)
        if tp.subject.type == 'VAR':
            var_pos[0] = 1
        if tp.predicate.type == 'VAR':
            var_pos[1] = 1
        if tp.object.type == 'VAR':
            var_pos[2] = 1
        
        #predicate features
        raw_predicate_freq_feats = np.zeros(4)
        cluster_bucket_feat = np.zeros(self.max_clusters)
        try:
            raw_predicate_freq_feats[0] = int(self.pred_freq[tp.predicate.node_label])
            unique_subj, unique_obj = self.pred_ents[tp.predicate.node_label]
            raw_predicate_freq_feats[3] = int(self.pred_lits[tp.predicate.node_label])
        except KeyError:
            raw_predicate_freq_feats[0] = 0
            unique_subj, unique_obj = 0,0
            raw_predicate_freq_feats[3] = 0
        
        raw_predicate_freq_feats[1] = int(unique_subj)
        raw_predicate_freq_feats[2] = int(unique_obj)
        try:
            pred_indices = self.pred2index[tp.predicate.node_label]
            for idx in pred_indices:
                cluster_bucket_feat[idx] = 1
        except KeyError:
            if self.verbose:
                print(f'Clustering error {tp.predicate.node_label} not in train log')
                
        
        res =  np.concatenate( (var_pos, raw_predicate_freq_feats, cluster_bucket_feat)).astype(np.float32)
        #return np.concatenate( (var_pos, raw_predicate_freq_feats, cluster_bucket_feat))
        return res
        
        
        if Node.use_join_features:
            join_feat = self.get_join_features()
        else:
            join_feat = np.array([])
        
        np.concatenate((join_feat, predicate_features))
    def create_cluster_from_pred_file(self, pred_file, save_pred_graph_png = None, community_no:int=10, verbose = False,  load_path=None, save_path=None):
        preds = get_bgp_predicates_from_path(pred_file)
        self.__create_clusters__(preds, save_pred_graph_png=save_pred_graph_png, community_no=community_no, verbose=verbose, load_path=load_path, save_path=save_path)
    
    def __print_communities(self, communities):
        
        for i, com in enumerate(communities):
            print("-"*25)
            print(f"Communitity {i} ")
            for node in com:
                print(node, end = '  ')
            print('\n')
            print("-"*25)
    
    def __create_clusters__(self,predicates : list[list[str]], save_pred_graph_png =None, community_no:int=10, verbose=False, load_path=None, save_path=None):
        if load_path != None:
            dat = json.load(open(load_path,'r'))
            self.pred2index = dat[0]
            self.max_clusters = dat[1]
            return
        
        pred_graph = nx.Graph()
        if verbose:
            print("-"*40+ '\n')
            print('Beginning Predicate Graph Construction'+'\n')
            print("-"*40)
        
        for bgp in predicates:
            pred_graph.add_nodes_from(bgp)
            for u in bgp:
                for v in bgp:
                    if u != v:
                        pred_graph.add_edge(u,v)
        
        if save_pred_graph_png != None:
            net = Network()
            net.from_nx(pred_graph)
            net.save_graph(save_pred_graph_png)
        if verbose:
            print("-"*40+ '\n')
            print('Beginning Clustering'+'\n')
            print("-"*40)
        communities = nx.community.girvan_newman(pred_graph)
        #print(next(communities))
        com_sets = []
        #com_sets = None
        for com in itertools.islice(communities, community_no):
            extracted_coms = tuple(sorted(c) for c in com)
            for no_com,extr_c in enumerate(extracted_coms):
                if verbose:
                    print(f'Working on Commnity {no_com}', end='\r')
                
                if not extr_c in com_sets:
                    com_sets.append(extr_c)
        if verbose:
            print("-"*40+ '\n')
            print('Beginning Cluster Postprocessing'+'\n')
            print("-"*40)
        pred_2_index = {}
        for idx, com in enumerate(com_sets):
            for pred in com:
                if pred in pred_2_index.keys():
                    pred_2_index[pred].append(idx)
                else:
                    pred_2_index[pred] = [idx]
        
        self.pred2index = pred_2_index
        self.max_clusters = len(com_sets)
        if save_path != None:
            json.dump([self.pred2index,self.max_clusters], open(save_path,'w'))
            return
if __name__ == "__main__":
    train_log = '/work/data/confs/May2/debug_train.json'
    featurizer = BaseFeaturizer( train_log = train_log, is_pred_clust_feat=True,save_pred_graph_png = None, community_no=10,path_pred_clust = { 'load_path':None, 'save_path':'/work/data/confs/May2/pred_clust.json'})
    bpgs_string = '{"[?x http://www.wikidata.org/prop/direct/P1936 ?z, ?y http://www.wikidata.org/prop/direct/P1652 ?x]": {"with_runtime": 421605504, "without_runtime": 290386128, "with_size": 0, "without_size": 0}}'
    bgp_dict = json.loads(bpgs_string)
    bgp_string = list(bgp_dict.keys())[0]
    info = bgp_dict[bgp_string]
    bgp = BGP(bgp_string, info)
    print(featurizer.get_feat_vec(bgp.triples[0]))