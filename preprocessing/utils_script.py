import json
import networkx as nx, numpy as np
import torch
from classifier.batched.trainer import get_debug_data_loader
from feature_check.entity_check import filtered_query_missing_entity
from feature_check.predicate_check import filter_missing_query_predicate
from feature_extraction.predicate_features_sub_obj import Predicate_Featurizer_Sub_Obj
from feature_extraction.predicates.predicate_features import PredicateFeaturesQuery
import os
import pickle as pcl
from feature_extraction.predicates.ql_pred_featurizer import ql_pred_featurizer
from graph_construction.nodes.node import Node
from graph_construction.bgp import BGP
from graph_construction.nodes.ql_node import ql_node
from graph_construction.triple_pattern import TriplePattern
from graph_construction.bgp_graph import BGPGraph
from glb_vars import PREDS_W_NO_BIN
import argparse, configparser
from feature_extraction.constants import PATH_TO_CONFIG, PATH_TO_CONFIG_GRAPH
from preprocessing.utils import get_predicates_from_path, ground_truth_distibution, load_BGPS_from_json
import sys
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

np.random.seed(42)

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

def get_predicate_freq_in_ql(bgps):
    pred_mapper = {}
    for bgp in bgps:
        bgp:BGP
        for triple in bgp.triples:
            triple:TriplePattern
            if triple.predicate.node_label in pred_mapper.keys():
                pred_mapper[triple.predicate.node_label] += 1
            else:
                pred_mapper[triple.predicate.node_label] = 1
    dct = {'predicate':[], 'freq':[]}
    for key in pred_mapper.keys():
        dct['predicate'].append(key)
        dct['freq'].append(pred_mapper[key])

    df_pred_id = pd.DataFrame.from_dict(dct)
    df_pred_id['freq'] = pd.to_numeric(df_pred_id['freq'])
    df_pred_id = df_pred_id.sort_values('freq',ascending=False)
    print(df_pred_id)
    print(df_pred_id.head(20))
    print(len(df_pred_id[(df_pred_id.freq < 10)]))
    print(df_pred_id.describe())
    exit()
    df_pred_id = df_pred_id.assign(row_number=range(len(df_freq)))

def load_bgps():
    parser = configparser.ConfigParser()
    #parser.read(PATH_TO_CONFIG)
    parser.read(PATH_TO_CONFIG_GRAPH)
    #bgps = load_BGPS_from_json(parser['Dataset']['train_data_path'])
    bgps = load_BGPS_from_json(parser['data_files']['val_data'])
    return bgps, parser

def stratified_split_preds( equalise_triples = False, limit=1100):
    bgps, parser = load_bgps()
    total_bgps = len(bgps)
    print(total_bgps)
    bgps = filter_missing_query_predicate(bgps,parser)
    print(f"Remaining bgps {len(bgps)} of {total_bgps}")
    df = pd.DataFrame(bgps ,columns=['bgp'])
    gts = [bgp.ground_truth for bgp in bgps]
    df['gt'] = gts
    
    #number_samples = int(parser['DebugDataset']['LIMIT'])
    number_samples = len(bgps)
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
    
    json.dump(bgps_to_dict(train_bgps), open(parser['DebugDataset']['train_data'], 'w'))
    json.dump(bgps_to_dict(val_bgps), open(parser['DebugDataset']['val_data'], 'w'))
    json.dump(bgps_to_dict(test_bgps), open(parser['DebugDataset']['test_data'], 'w')) 
    

def bgps_to_dict(bgps : list[BGP]):
    j = {}
    for x in bgps:
        j[x.bgp_string] = x.data_dict
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


def get_bin_topk_dict_rdf(PRED_FEATURIZER,feat_generation_path,bin_no, topk, NODE, parser):
    pred_feature_rizer:Predicate_Featurizer_Sub_Obj = PRED_FEATURIZER.prepare_pred_featues_for_bgp(feat_generation_path, bins=bin_no, topk=topk, obj_type=PRED_FEATURIZER)
    data_file = parser['DebugDataset']['train_data']
    data_file = parser['Dataset']['train_data_path']
    data_file = parser['data_files']['val_data']
    train_bgps = load_BGPS_from_json(data_file, node=NODE)
    pred_buckets = {}
    pred_topk = {}
    for bgp in train_bgps:
        bgp:BGP
        for t in bgp.triples:
            t:TriplePattern
            top = pred_feature_rizer.top_k_predicate(t.predicate.node_label)
            if top in pred_topk.keys():
                pred_topk[top].append(top)
            else:
                pred_topk[top] = [top]
            bucket= pred_feature_rizer.get_bin(t.predicate.node_label)
            if bucket in pred_buckets.keys():
                pred_buckets[bucket].append(bucket)
            else:
                pred_buckets[bucket] = [bucket]
    return pred_buckets,pred_topk

def get_bin_topk_dict_ql(PRED_FEATURIZER,feat_generation_path,bin_no, topk, NODE, parser):
    pred_feature_rizer:ql_pred_featurizer = PRED_FEATURIZER.prepare_pred_featues_for_bgp(feat_generation_path, bins=bin_no, topk=topk, obj_type=PRED_FEATURIZER)
    data_file = parser['DebugDataset']['train_data']
    data_file = parser['Dataset']['train_data_path']
    preds = get_predicates_from_path(data_file)
    pred_feature_rizer.prepare_featurizer(preds,topk,bin_no)
    
    train_bgps = load_BGPS_from_json(data_file, node=Node)
    
    pred_buckets = {}
    pred_topk = {}
    for bgp in train_bgps:
        bgp:BGP
        for t in bgp.triples:
            t:TriplePattern
            top = pred_feature_rizer.top_k_predicate(t.predicate.node_label)
            if top in pred_topk.keys():
                pred_topk[top].append(top)
            else:
                pred_topk[top] = [top]
            bucket= pred_feature_rizer.get_bin(t.predicate.node_label)
            if bucket in pred_buckets.keys():
                pred_buckets[bucket].append(bucket)
            else:
                pred_buckets[bucket] = [bucket]
    return pred_buckets,pred_topk

#return the amount of used buckets.
def plot_bucket_stat(bucket_dct,bin_no, path='/work/data/confs/May2/pred_charts', color='green') -> int:
    dct = {'buckets':[], 'Count':[]}
    for key in bucket_dct.keys():
        dct['buckets'].append(key)
        dct['Count'].append(len(bucket_dct[key]))
    df = pd.DataFrame.from_dict(dct)
    df = df.sort_values(by=['Count'],ascending=False)
    plt.clf()
    plt.barh([f"bucket_{x}" for x in list(df['buckets'])], list(df['Count']), 0.75, color =color)
    plt.xlabel('Occurences of Buckets')
    plt.ylabel("Bucket Identifiers")
    plt.title(f"Occurence Plot for Predicate Bucket ({bin_no})")
    for i in range(len(df)):
        plt.text(df['Count'].iloc[i]+2.5, i+0.0, str(df['Count'].iloc[i]))
    #plt.bar(range(len(df)), list(df['Count']), align='center')
    #plt.xticks(range(len(df)), list(df['buckets']))
    plt.xscale('log')
    plt.subplots_adjust(bottom=0.1,top=0.95,hspace=0, wspace=0, right=0.95, left=0.2)
    
    plt.savefig(f"{path}/buckets_{bin_no}.png")
    plt.close()
    return len(dct['buckets'])
    
    
    
def pred_bucket_analysis_rdf():
    parser = configparser.ConfigParser()
    parser.read(PATH_TO_CONFIG_GRAPH)
    
    NODE = Node
    PRED_FEATURIZER = Predicate_Featurizer_Sub_Obj
    feat_generation_path = parser['PredicateFeaturizerSubObj']['load_path']
    topk = int(parser['PredicateFeaturizerSubObj']['topk'])
    bin_no = int(parser['PredicateFeaturizerSubObj']['bin_no'])
    bin_no = 30
    max_bucket_size, max_bin_no = 0,0
    for bin_no in range(20, 50):
        pred_buckets,_ = get_bin_topk_dict_rdf(PRED_FEATURIZER,feat_generation_path,bin_no, topk, NODE, parser)
        #print("Bucket stats")
        #for key in pred_buckets.keys():
        #    print(f"{key}: {len(pred_buckets[key])}")
        print(f"Starting plottin bucket no {bin_no} ...")
        bucket_size = plot_bucket_stat(pred_buckets,bin_no)
        if bucket_size > max_bucket_size:
            max_bucket_size = bucket_size
            max_bin_no = bin_no
        print(f"Finished plottin bucket no {bin_no}!")
    print(f"Binning with bucket size {max_bin_no} has most used bucket [{max_bucket_size}]")
    
    exit()
    pred_buckets,pred_topk = get_bin_topk_dict_rdf(PRED_FEATURIZER,feat_generation_path,bin_no, topk, NODE, parser)
    print("topk stats")
    for key in pred_topk.keys():
        print(f"{key}: {len(pred_topk[key])}")
    print("Bucket stats")
    for key in pred_buckets.keys():
        print(f"{key}: {len(pred_buckets[key])}")
    #train_loader, val_loader, test_loader = get_data_loader(parser)
    #train_loader = torch.load('/work/data/confs/April25/train_dataset.pickle')
    #for batch in train_loader:
    #    print(batch)
    #    exit()
    pass  
def pred_bucket_analysis_querylog(path='/work/data/confs/May2/query_log_preds'):
    parser = configparser.ConfigParser()
    parser.read(PATH_TO_CONFIG_GRAPH)
    
    NODE = ql_node
    PRED_FEATURIZER = ql_pred_featurizer
    feat_generation_path = parser['PredicateFeaturizerSubObj']['load_path']
    topk = int(parser['PredicateFeaturizerSubObj']['topk'])
    bin_no = int(parser['PredicateFeaturizerSubObj']['bin_no'])
    bin_no = 30
    max_bucket_size, max_bin_no = 0,0
    for bin_no in range(5, 50):
        pred_buckets,_ = get_bin_topk_dict_ql(PRED_FEATURIZER,feat_generation_path,bin_no, topk, NODE, parser)
        #print("Bucket stats")
        #for key in pred_buckets.keys():
        #    print(f"{key}: {len(pred_buckets[key])}")
        print(f"Starting plottin bucket no {bin_no} ...")
        bucket_size = plot_bucket_stat(pred_buckets,bin_no, path=path, color='blue')
        if bucket_size > max_bucket_size:
            max_bucket_size = bucket_size
            max_bin_no = bin_no
        print(f"Finished plottin bucket no {bin_no}!")
    print(f"Binning with bucket size {max_bin_no} has most used bucket [{max_bucket_size}]")
    
    exit()
    
def bgps_duplicate_checker():
    parser = configparser.ConfigParser()
    #parser.read(PATH_TO_CONFIG)
    parser.read(PATH_TO_CONFIG_GRAPH)
    data_file = parser['DebugDataset']['train_data']
    data_file = parser['Dataset']['train_data_path']
    bgps = load_BGPS_from_json(data_file)
    
    for bgp in bgps:
        bgp:BGP
        bgp.triples = set(bgp.triples)
    
    for i in range(len(bgps)):
        print(f"Iterating {i} ...", end='\r')
        for j in range(i+1, len(bgps)):
            if i != j and len(bgps[i].triples) == len(bgps[j].triples):
                if bgps[i].triples == bgps[j].triples:
                    yield bgps[i]

def bgps_stats():
    parser = configparser.ConfigParser()
    #parser.read(PATH_TO_CONFIG)
    parser.read(PATH_TO_CONFIG_GRAPH)
    data_file = parser['DebugDataset']['train_data']
    data_file = parser['data_files']['val_data']
    bgps = load_BGPS_from_json(data_file)
    
    total_triples = []
    for bgp in bgps:
        bgp:BGP
        total_triples.append( len(bgp.triples))
    total_triples = np.array(total_triples)
    unique_vals = np.unique( total_triples)
    print(f"BGP stats for {data_file}")
    for val in unique_vals:
        count = np.count_nonzero(total_triples==val)
        print(f"\tFraction with {val}: {count/len(total_triples)}, Count: {count}")
        
    exit()
    print(f"\tMean triples: {total_triples.mean()}")
    print(f"\tSTD triples: {np.std(total_triples)}")
    print(f"\t25% triples: {np.quantile( total_triples, q=0.25)}")
    print(f"\t50% triples: {np.quantile( total_triples, q=0.50)}")
    print(f"\t75% triples: {np.quantile( total_triples, q=0.75)}")
    print(f"\tmax triples: {np.max( total_triples)}")
    exit()

def temp():
    parser = configparser.ConfigParser()
    parser.read(PATH_TO_CONFIG_GRAPH)
    
    NODE = ql_node
    PRED_FEATURIZER = ql_pred_featurizer
    feat_generation_path = parser['PredicateFeaturizerSubObj']['load_path']
    topk = int(parser['PredicateFeaturizerSubObj']['topk'])
    bin_no = int(parser['PredicateFeaturizerSubObj']['bin_no'])
    bin_no = 30
    
    pred_feature_rizer:ql_pred_featurizer = PRED_FEATURIZER.prepare_pred_featues_for_bgp(feat_generation_path, bins=bin_no, topk=topk, obj_type=PRED_FEATURIZER)
    data_file = parser['DebugDataset']['train_data']
    data_file = parser['Dataset']['train_data_path']
    preds = get_predicates_from_path(data_file)
    print(f"amount of predicates {len(preds)}")
    exit()
    pred_feature_rizer.prepare_featurizer(preds,topk,bin_no)
    
    train_bgps = load_BGPS_from_json(data_file, node=Node)
    
    pred_buckets = {}
    pred_topk = {}
    for bgp in train_bgps:
        bgp:BGP
        for t in bgp.triples:
            t:TriplePattern
            top = pred_feature_rizer.top_k_predicate(t.predicate.node_label)
            if top in pred_topk.keys():
                pred_topk[top].append(top)
            else:
                pred_topk[top] = [top]
            bucket= pred_feature_rizer.get_bin(t.predicate.node_label)
            if bucket in pred_buckets.keys():
                pred_buckets[bucket].append(bucket)
            else:
                pred_buckets[bucket] = [bucket]
    return pred_buckets,pred_topk
if __name__ == "__main__":
    #stratified_split()
    if len(sys.argv) > 1 and sys.argv[1] == 'split':
        stratified_split_v2()
    elif len(sys.argv) > 1 and sys.argv[1] == 'stat':
        data_stats()
    elif len(sys.argv) > 1 and sys.argv[1] == 'dataset_stat':
        pred_bucket_analysis_rdf()
    elif len(sys.argv) > 1 and sys.argv[1] == 'ql_pred_buckets':
        pred_bucket_analysis_querylog()
    elif len(sys.argv) > 1 and sys.argv[1] == 'stratified_split':
        stratified_split_preds()
    else:
        #dup_gen = bgps_duplicate_checker()
        #print(next(dup_gen))
        bgps_stats()
        exit()
        temp()
        