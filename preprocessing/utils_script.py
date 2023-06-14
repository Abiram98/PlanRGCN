import json
import numpy as np
from feature_check.predicate_check import filter_missing_query_predicate
from feature_extraction.predicate_features_sub_obj import Predicate_Featurizer_Sub_Obj

from feature_extraction.predicates.ql_pred_featurizer import ql_pred_featurizer
from graph_construction.nodes.node import Node
from graph_construction.bgp import BGP
from graph_construction.nodes.ql_node import ql_node
from graph_construction.triple_pattern import TriplePattern
import argparse, configparser
from feature_extraction.constants import PATH_TO_CONFIG_GRAPH
from preprocessing.utils import convert_leaf_to_json, get_predicates_from_path, ground_truth_distibution, load_BGPS_from_json
import matplotlib.pyplot as plt
import pandas as pd
from preprocessing.split_utils import stratified_split_v2,stratified_split_preds

np.random.seed(42)

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
def single_re(rt_jena, rt_bloom, mean_jena):
    return (abs(rt_jena - rt_bloom) )/mean_jena

def single_re(rt_jena, rt_bloom, mean_jena):
    #return (abs(rt_jena - rt_bloom) )/mean_jena
    return (abs(rt_jena - rt_bloom) )/rt_jena
    return (abs(rt_jena - rt_bloom) )/rt_jena
#deprecated relative error
def relative_error(truth, pred, aggregate = True):
        truth = truth
        pred = pred
        #act_mean = np.mean(truth)
        #print('Jena Mean: ',act_mean)
        #r_e = [(abs(act_i - pred_i) )/act_mean for act_i, pred_i in zip (truth, pred)]
        r_e = [(abs(act_i - pred_i) )/act_i for act_i, pred_i in zip (truth, pred)]
        
        if aggregate:
            return np.sum(r_e)/len(r_e)
        else:
            return r_e
#use this one
def relative_error_v2(pred, gt):
    return (gt- pred)/gt

def re_gt_analysis(bgps: dict, threshold, mean_jena):
    ones, zeros = 0,0
    for k in bgps.keys():
        if relative_error_v2(int(bgps[k]['jena_runtime']),int(bgps[k]['bloom_runtime']),mean_jena) > threshold:
            ones += 1
        else:
            zeros += 1
    return ones, zeros
def process_gt(path, output, gt_type= 're', th=None):
    bgps = json.load(open(path, 'r'))
    
    if gt_type == 'std':
        rt, rt_without = [], []
        for k in bgps.keys():
            rt.append(int(bgps[k]['bloom_runtime'])) #with bloom filter
            rt_without.append(int(bgps[k]['jena_runtime'])) #without BloomFilter
        wo_std = np.std(rt_without)
        w_std = np.std(rt)
        print(wo_std, w_std)
        for k in bgps.keys():
            if bgps[k]['bloom_runtime']+w_std < bgps[k]['bloom_runtime'] - wo_std:
                bgps[k]['gt'] = True
            else:
                bgps[k]['gt'] = False
            """if bgps[k]['with_runtime'] > bgps[k]['without_runtime']:
                if bgps[k]['with_runtime']-w_std > bgps[k]['without_runtime'] + wo_std:
                    bgps[k]['gt'] = False
                else:
                    bgps[k]['gt'] = True
            elif bgps[k]['without_runtime'] > bgps[k]['with_runtime']:
                if bgps[k]['without_runtime']- wo_std > bgps[k]['with_runtime']+w_std :
                    bgps[k]['gt'] = True
                else:
                    bgps[k]['gt'] = False"""
    elif gt_type == 're':
        assert th != None
        th = float(th)
        bgps = gt_re_assignemnt(bgps, th)
    elif gt_type == 'median':
        bgps = gt_median_assignment(bgps)
    else:
        print(f'Please provide a legal Ground Truth assignment type. "{gt_type}" is not legal.')
        exit()
    json.dump(bgps, open(output,'w'))

def gt_median_assignment(bgps, runtime_field='jena_runtime'):
    n_bgps = {}
    rt = []
    for k in bgps.keys():
        rt.append(int(bgps[k][runtime_field])) #without BloomFilter
    median = np.median(rt)
    print(f"Median for queries {median}")
    ones, zeros = 0,0
    for k in bgps.keys():
        temp_dct = bgps[k]
        temp_dct['gt'] = True if temp_dct[runtime_field] >= median else False
        if temp_dct['gt'] == True:
            ones += 1
        else:
            zeros += 1
        n_bgps[k] = temp_dct
    print(f'Distribution of GT: 1:{ones} [{ones/(ones+zeros)}] 0:{zeros} [{zeros/(ones+zeros)}]')    
    return n_bgps

def gt_re_assignemnt(bgps, threshold= 0.1):
    n_bgps = {}
    rt = []
    for k in bgps.keys():
        rt.append(int(bgps[k]['jena_runtime']))
    mean_jena = np.mean(rt)
    ones, zeros = 0,0
    for k in bgps.keys():
        n_bgps[k] = bgps[k]
        if relative_error_v2(int(bgps[k]['bloom_runtime']),int(bgps[k]['jena_runtime'])) >= threshold:
            n_bgps[k]['gt'] = True
            ones += 1
        else:
            n_bgps[k]['gt'] = False
            zeros += 1
    print(f'GT distribution: zero {zeros} [{zeros/(zeros+ones)}], ones {ones} [{ones/(zeros+ones)}] with threshold {threshold}')
    return n_bgps

def triple_stat( input_path):
    #bgps = json.load(open(input_path, 'r'))
    bgps=load_BGPS_from_json(input_path)
    trpl = {}
    for b in bgps:
        if len(b.triples) in trpl.keys():
            trpl[len(b.triples)] +=1
        else:
            trpl[len(b.triples)] = 1
    print(f"Triple count Statistics:")
    keys = sorted(list(trpl.keys()))
    for k in keys:
        print(f"\t{k}: {trpl[k]} [{trpl[k]/len(bgps)}]")

def get_runtimes(bgps,runtime_field='jena_runtime' , normalise = False):
    rt = []
    for k in bgps.keys():
        rt.append(int(bgps[k][runtime_field])) #without BloomFilter
    if normalise:
        rt = [(r*1e-9) for r in rt]
    
    return rt

def print_latency_stats(path, runtime_field='jena_runtime'):
    bgps = json.load(open(path, 'r'))
    rt = get_runtimes(bgps,runtime_field=runtime_field )
    print(f'Statistics for {path}')
    print(f'\tMedian: {np.median(rt)}')
    print(f'\tStd: {np.std(rt)}')
    print(f'\tAverage: {np.mean(rt)}')
    print(f'\t25%-quantile: {np.quantile(rt,q=0.25)}')
    print(f'\t75%-quantile: {np.quantile(rt,q=0.75)}')
    #print(json.dumps(rt))


if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser(prog='Util scripts')
    arg_parse.add_argument('task')
    arg_parse.add_argument('--val_path', '--val_path')
    arg_parse.add_argument('--test_path', '--test_path')
    arg_parse.add_argument('--input', '--input')
    arg_parse.add_argument('--output', '--output')
    arg_parse.add_argument('--time_out', '--time_out')
    arg_parse.add_argument('--gt_type', '--gt_type')
    arg_parse.add_argument('--threshold', '--threshold')
    args = arg_parse.parse_args()
    val_path, test_path = args.val_path, args.test_path
    output = args.output
    #stratified_split()
    if args.task == 'split':
        stratified_split_v2()
    elif args.task == 'stat':
        data_stats()
    elif args.task == 'dataset_stat':
        pred_bucket_analysis_rdf()
    elif args.task == 'ql_pred_buckets':
        pred_bucket_analysis_querylog()
    elif args.task == 'stratified_split':
        #stratified_split_preds(equalise_triples=True)
        stratified_split_preds(input_path=args.input, output=args.output ,equalise_gt=True)
    elif args.task == 'gt_assign':
        process_gt(args.input, output,gt_type=args.gt_type, th=args.threshold)
    elif args.task == 'convert_leaf':
        convert_leaf_to_json(args.input, output, time_out=args.time_out)
    elif args.task == 'triple_stat':
        triple_stat(args.input)
    elif args.task == 'lat_stat':
        print_latency_stats(args.input)
    else:
        #dup_gen = bgps_duplicate_checker()
        #print(next(dup_gen))
        bgps_stats()
        exit()
        temp()
        