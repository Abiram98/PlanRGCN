import json
import networkx as nx, numpy as np
from feature_check.entity_check import filtered_query_missing_entity
from feature_check.predicate_check import filter_missing_query_predicate
from feature_extraction.predicate_features import PredicateFeaturesQuery
import os
import pickle as pcl
from graph_construction.node import Node
from graph_construction.bgp import BGP
from graph_construction.triple_pattern import TriplePattern
from graph_construction.bgp_graph import BGPGraph
from glb_vars import PREDS_W_NO_BIN
import argparse, configparser
from feature_extraction.constants import PATH_TO_CONFIG
from utils import load_BGPS_from_json

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
    
def bgps_to_dict(bgps : list[BGP]):
    j = {}
    for x in bgps:
        j[x.bgp_string] = x.ground_truth
    return j
if __name__ == "__main__":
    stratified_split()