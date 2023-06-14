import json
import os
import networkx as nx, numpy as np
from feature_check.entity_check import filtered_query_missing_entity
from feature_check.predicate_check import filter_missing_query_predicate
from feature_extraction.predicate_features_sub_obj import Predicate_Featurizer_Sub_Obj

from feature_extraction.predicates.ql_pred_featurizer import ql_pred_featurizer
from graph_construction.nodes.node import Node
from graph_construction.bgp import BGP
from graph_construction.nodes.ql_node import ql_node
from graph_construction.triple_pattern import TriplePattern
import argparse, configparser
from feature_extraction.constants import PATH_TO_CONFIG, PATH_TO_CONFIG_GRAPH
from preprocessing.utils import convert_leaf_to_json, get_predicates_from_path, ground_truth_distibution, load_BGPS_from_json
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

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
    bgps, parser = load_bgps()
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
        j[x.bgp_string] = x.data_dict
    return j

def load_bgps(input_path=None):
    parser = configparser.ConfigParser()
    #parser.read(PATH_TO_CONFIG)
    parser.read(PATH_TO_CONFIG_GRAPH)
    #bgps = load_BGPS_from_json(parser['Dataset']['train_data_path'])
    if input_path != None:
        bgps = load_BGPS_from_json(input_path)
        count = 0
        for x in bgps:
            if x.ground_truth == 0:
                count += 1
        print('here',count)
    else:
        bgps = load_BGPS_from_json(parser['data_files']['combined'])
    return bgps, parser

def stratified_split_preds(input_path=None, output=None, equalise_gt = False, limit=1100):
    
    bgps, parser = load_bgps(input_path=input_path)
    total_bgps = len(bgps)
    #bgps = filter_missing_query_predicate(bgps,parser)
    #print(f"Remaining bgps {len(bgps)} of {total_bgps}")
    df = pd.DataFrame(bgps ,columns=['bgp'])
    gts = [bgp.ground_truth for bgp in bgps]
    df['gt'] = gts
    if equalise_gt:
        uniq = df['gt'].unique()
        dfs = []
        min_els = None
        for x in uniq:
            temp = df[df['gt'] == x]
            if min_els == None or min_els > len(temp):
                min_els = len(temp)
                print('temp', len(temp))
            dfs.append(  temp)
        print(min_els)
        dfs2 = []
        for d in dfs:
            d = d.sample(frac=1).reset_index(drop=True)
            d = d.head(min_els)
            dfs2.append(d)
        df = pd.concat(dfs2)
    
    #number_samples = int(parser['DebugDataset']['LIMIT'])
    #number_samples = len(bgps)
    number_samples = len(df)
    train_per,  test_perc = 0.8, 0.2
    train_no, test_no = round(train_per*number_samples),round(test_perc*number_samples)    
    
    X_train, X_test, y_train, _ = train_test_split(df, df[['gt']], test_size=test_no, train_size=train_no)
    X_train, X_val, y_train, _ = train_test_split(X_train, y_train, test_size=test_no)
    
    train_bgps = X_train['bgp'].tolist()
    val_bgps = X_val['bgp'].tolist()
    test_bgps = X_test['bgp'].tolist()
    print(f"Train {len(train_bgps)}, Val {len(val_bgps)}, Test {len(test_bgps)}")

    print("Train: Distribution of ground truth")
    ground_truth_distibution(train_bgps, verbose=True)
    print("Val: Distribution of ground truth")
    ground_truth_distibution(val_bgps, verbose=True)
    print("Test: Distribution of ground truth")
    ground_truth_distibution(test_bgps, verbose=True)
    if output == None:
        json.dump(bgps_to_dict(train_bgps), open(parser['DebugDataset']['train_data'], 'w'))
        json.dump(bgps_to_dict(val_bgps), open(parser['DebugDataset']['val_data'], 'w'))
        json.dump(bgps_to_dict(test_bgps), open(parser['DebugDataset']['test_data'], 'w'))
    else:
        if not os.path.isdir(output):
            print(f"ERROR: Output should be a folder but was {output}")
            exit()
        json.dump(bgps_to_dict(train_bgps), open(f"{output}/train.json", 'w'))
        json.dump(bgps_to_dict(val_bgps), open(f"{output}/val.json", 'w'))
        json.dump(bgps_to_dict(test_bgps), open(f"{output}/test.json", 'w')) 