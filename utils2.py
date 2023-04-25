import json
import networkx as nx, numpy as np
from feature_check.entity_check import filtered_query_missing_entity
from feature_check.predicate_check import filter_missing_query_predicate
from feature_extraction.predicate_features import PredicateFeaturesQuery
import os
import pickle as pcl
from graph_construction.nodes.node import Node
from graph_construction.bgp import BGP
from graph_construction.triple_pattern import TriplePattern
from graph_construction.bgp_graph import BGPGraph
from glb_vars import PREDS_W_NO_BIN
import argparse, configparser
from feature_extraction.constants import PATH_TO_CONFIG, PATH_TO_CONFIG_GRAPH
from utils import ground_truth_distibution, load_BGPS_from_json
import sys
from sklearn.metrics import jaccard_score
def stratified_split():
    global bgps
    parser = configparser.ConfigParser()
    parser.read(PATH_TO_CONFIG)
    bgps = load_BGPS_from_json('/work/data/train_data.json')
    total_bgps = len(bgps)
    bgps = filtered_query_missing_entity(parser)
    bgps = filter_missing_query_predicate(bgps,parser)
    print(f"Remaining bgps {len(bgps)} of {total_bgps}")
    train_per, val_perc, test_perc = 0.6, 0.2, 0.2
    
    train_n, test_n = round(train_per*len(bgps)),round(test_perc*len(bgps))
    
    train_bgps = bgps[:train_n]
    val_bgps = bgps[train_n:]
    test_bgps = val_bgps[test_n :]
    val_bgps = val_bgps[:test_n]
    
    print(f"Train {len(train_bgps)}, Val {len(val_bgps)}, Test {len(test_bgps)}")
    
    json.dump(bgps_to_dict(train_bgps), open(parser['DebugDataset']['train_data'], 'w'))
    json.dump(bgps_to_dict(val_bgps), open(parser['DebugDataset']['val_data'], 'w'))
    json.dump(bgps_to_dict(test_bgps), open(parser['DebugDataset']['test_data'], 'w'))


def stratified_split_v2():
    parser = configparser.ConfigParser()
    #parser.read(PATH_TO_CONFIG)
    parser.read(PATH_TO_CONFIG_GRAPH)
    bgps = load_BGPS_from_json(parser['Dataset']['train_data_path'])
    total_bgps = len(bgps)
    print(total_bgps)
    bgps = filter_missing_query_predicate(bgps,parser)
    print(f"Remaining bgps {len(bgps)} of {total_bgps}")
    
    number_samples = int(np.floor(int(parser['DebugDataset']['LIMIT'])/2))
    bgps_ones = [x for x in bgps if x.ground_truth == 1]
    bgps_ones = bgps_ones[:number_samples]
    bgps_zeros = [x for x in bgps if x.ground_truth == 0]
    bgps_zeros = bgps_zeros[:number_samples]
    #bgps = bgps[:int(parser['DebugDataset']['LIMIT'])]
    
    train_per, val_perc, test_perc = 0.6, 0.2, 0.2
    
    train_n_ones, test_n_ones = round(train_per*len(bgps_ones)),round(test_perc*len(bgps_ones))
    train_n_zeros, test_n_zeroes = round(train_per*len(bgps_zeros)),round(test_perc*len(bgps_zeros))
    
    train_bgps = bgps_ones[:train_n_ones]
    train_bgps.extend(bgps_zeros[:train_n_zeros])
    
    val_bgps = bgps_ones[train_n_ones:-test_n_ones]
    val_bgps.extend(bgps_zeros[train_n_zeros:-test_n_zeroes])
    
    test_bgps = bgps_ones[train_n_ones +test_n_ones:]
    test_bgps.extend(bgps_zeros[train_n_zeros+test_n_ones:])
    
    #val_bgps = val_bgps[:test_n_ones]
    
    print(f"Train {len(train_bgps)}, Val {len(val_bgps)}, Test {len(test_bgps)}")
    print("Train: Distribution of ground truth")
    ground_truth_distibution(train_bgps, verbose=True)
    print("Val: Distribution of ground truth")
    ground_truth_distibution(val_bgps, verbose=True)
    print("Test: Distribution of ground truth")
    ground_truth_distibution(test_bgps, verbose=True)
    
    json.dump(bgps_to_dict(train_bgps), open(parser['DebugDataset']['train_data'], 'w'))
    json.dump(bgps_to_dict(val_bgps), open(parser['DebugDataset']['val_data'], 'w'))
    json.dump(bgps_to_dict(test_bgps), open(parser['DebugDataset']['test_data'], 'w'))   

def bgps_to_dict(bgps : list[BGP]):
    j = {}
    for x in bgps:
        j[x.bgp_string] = x.ground_truth
    return j

def jaccard(list1, list2):
    intersection = len(list(set(list1).intersection(list2)))
    union = len(list(set(list1).union(list2)))
    return float(intersection) / union

def data_stats():
    parser = configparser.ConfigParser()
    #parser.read(PATH_TO_CONFIG)
    parser.read(PATH_TO_CONFIG_GRAPH)
    train_preds = set()
    val_preds = set()
    test_preds = set()
    for i in ['train_data','val_data', 'test_data']:
        print(f'For {i}')
        bgps = load_BGPS_from_json(parser['DebugDataset'][i])
        total_bgps = len(bgps)
        bgps = filter_missing_query_predicate(bgps,parser)
        print(f"Remaining bgps {len(bgps)} of {total_bgps}")
        ground_truth_distibution(bgps, verbose=True)
        if i == 'train_data':
            for bgp in bgps:
                bgp:BGP
                for t in bgp.triples:
                    train_preds.add(t.predicate.node_label)
        if i == 'val_data':
            for bgp in bgps:
                bgp:BGP
                for t in bgp.triples:
                    val_preds.add(t.predicate.node_label)
        if i == 'test_data':
            for bgp in bgps:
                bgp:BGP
                for t in bgp.triples:
                    test_preds.add(t.predicate.node_label)
    print(f"Pred jarcard between train and validation: {jaccard(train_preds,val_preds)}")
    print(f"Pred jarcard between train and test: {jaccard(train_preds,test_preds)}")
    
    
if __name__ == "__main__":
    #stratified_split()
    if sys.argv[1] == 'split':
        stratified_split_v2()
    elif sys.argv[1] == 'stat':
        data_stats()