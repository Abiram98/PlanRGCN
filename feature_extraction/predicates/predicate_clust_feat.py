

import configparser
from feature_extraction.constants import PATH_TO_CONFIG_GRAPH
from feature_extraction.predicate_features_sub_obj import Predicate_Featurizer_Sub_Obj
import pandas as pd

from graph_construction.bgp import BGP
from preprocessing.utils import get_bgp_predicates_from_path, load_BGPS_from_json
import networkx as nx
import matplotlib.pyplot as plt
from pyvis.network import Network
import pickle
import time
import itertools
class Pred_clust_feat(Predicate_Featurizer_Sub_Obj):
    def __init__(self, endpoint_url=None, timeout=30):
        super().__init__(endpoint_url, timeout)
    
    #this function will be invoked during the loading of the function.
    def prepare_pred_feat(self, bins = 30, k=20):
        return
    
    #This needs to implemented for predicates.
    def get_pred_feat(self, pred_label):
        try:
            return self.pred2index[pred_label]
        except KeyError:
            return []

    def print_communities(self, communities, pred_graph):
        
        for i, com in enumerate(communities):
            print("-"*25)
            print(f"Communitity {i} ")
            for node in com:
                print(node, end = '  ')
            print('\n')
            print("-"*25)
        
    def create_cluster_from_pred_file(self, pred_file, save_pred_graph_png = None):
        preds = get_bgp_predicates_from_path(pred_file)
        self.create_clusters(preds, save_pred_graph_png=save_pred_graph_png)
    
    def create_clusters(self,predicates : list[list[str]], save_pred_graph_png =None, community_no:int=10, verbose=False):
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
        
        print(f"Amount of component in input graph: {nx.number_connected_components(pred_graph)}")
        exit()
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
        """print("-"*25+ '\n'*2)
        print('Beginning Stats'+'\n'*2)
        print("-"*25)
        zeros,ones, more = 0, 0, 0
        for k,v in pred_2_index.items():
            
            if len(v) == 1:
                ones += 1
            elif len(v) == 0:
                zeros +=1
            else:
                more +=1
        print(zeros, ones, more)
        print("nodes in graph and predicates in table:",len(pred_graph.nodes), len(pred_2_index.keys()))"""


if __name__ == "__main__":
    featurizer = Pred_clust_feat()
    parser = configparser.ConfigParser()
    parser.read(PATH_TO_CONFIG_GRAPH)
    #data_file = parser['Dataset']['train_data_path']
    data_file = parser['DebugDataset']['train_data']
    #bgps = load_BGPS_from_json(data_file)
    #bgps = get_bgp_predicates_from_path(data_file)
    #featurizer.create_clusters(bgps, save_pred_graph_png='pred_graph_debug.html')
    featurizer.create_cluster_from_pred_file(data_file)
    for k in featurizer.pred2index.keys():
        print(featurizer.pred2index[k])
        break
    #featurizer.create_clusters(bgps)
    #featurizer.create_clusters(bgps, save_pred_graph='train_pred_graph.html')
    #pickle.dump(featurizer,open('pred_clusterer.pickle', 'wb'))