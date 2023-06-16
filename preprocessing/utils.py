import json
import networkx as nx, numpy as np
#from feature_check.entity_check import filtered_query_missing_entity
#from feature_check.predicate_check import filter_missing_query_predicate
from feature_extraction.predicates.predicate_features import PredicateFeaturesQuery
import os
import pickle as pcl
from graph_construction.nodes.node import Node
from graph_construction.bgp import BGP
from graph_construction.nodes.node_num_pred import Node_num_pred_encoding
#from graph_construction.tps_graph import tps_graph
from graph_construction.triple_pattern import TriplePattern
from graph_construction.bgp_graph import BGPGraph
from glb_vars import PREDS_W_NO_BIN
import argparse, configparser
from feature_extraction.constants import PATH_TO_CONFIG
import time
from graph_construction.triple_pattern import get_tp_class


def load_BGPS_from_json(path,limit_bgp=None, node=Node, TP_graph_type='tp'):
    #pred_feat = None
    #if not path_predicate_feat_gen == None:
    #    pred_feat = PredicateFeaturesQuery.prepare_pred_featues_for_bgp(path_predicate_feat_gen, bins=bins)
    triple_class = get_tp_class(TP_graph_type)
    data = None
    with open(path,'rb') as f:
        data = json.load(f)
                
    if data == None:
        print('Data could not be loaded!')
        return
    
    BGP_strings = list(data.keys())
    if limit_bgp != None:
        BGP_strings = BGP_strings[:limit_bgp]
    BGPs = []
    for bgp_string in BGP_strings:
        start = time.time()
        
        b = BGP(bgp_string, data[bgp_string],node_class=node, TP_class=triple_class)
        duration = time.time() -start
        b.data_dict['bgp_construction_duration'] = duration
        BGPs.append(b)
    return BGPs

def get_predicates_from_path(path):
    data = None
    with open(path,'rb') as f:
        data = json.load(f)
                
    if data == None:
        print('Data could not be loaded!')
        return
    
    BGP_strings = list(data.keys())
    
    preds = []
    for bgp_string in BGP_strings:
        triple_strings = bgp_string[1:-1].split(',')
        for triple_string in triple_strings:
            splits = triple_string.split(' ')
            splits = [s for s in splits if s != '']
            if not splits[1].startswith('?'):
               preds.append(splits[1])
    return preds



def get_predicates(bgps: list):
    predicates = set()
    for bgp in bgps:
        for triple in bgp.triples:
            if triple.predicate.type == 'URI':
                predicates.add(triple.predicate)
    return list(predicates)

def get_entities(bgps: list):
    entities = set()
    for bgp in bgps:
        for triple in bgp.triples:
            for e in [triple.subject, triple.object]:
                if e.type == 'URI':
                    entities.add(e)
    return list(entities)



#Assumes a PREDS_W_NO_BIN global variable
def get_predicate_with_no_bin():
    preds = set()
    for x in PREDS_W_NO_BIN:
        preds.add(str(x))
    return list(preds), len(preds)

def ground_truth_distibution(bgps, verbose= False):
    ground_truth_1 = 0
    ground_truth_0 = 0
    for bgp in bgps:
        if bgp.ground_truth == 1:
            ground_truth_1 += 1
        elif bgp.ground_truth == 0:
            ground_truth_0 += 1
    if verbose:
        print(f"Ground truth distribtution:\n\t1: {ground_truth_1}/{ground_truth_1+ground_truth_0}, {ground_truth_1/(ground_truth_1+ground_truth_0)}")
        print(f"\t0: {ground_truth_0}/{ground_truth_1+ground_truth_0}, {ground_truth_0/(ground_truth_1+ground_truth_0)}")
    return ground_truth_0, ground_truth_1
    
def unpickle_obj(path):
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            return pcl.load(f)
    return None

def pickle_obj(obj,path):
    with open(path, 'wb') as f:
        pcl.dump(obj,f)
def numpy_to_string(nparr):
    return nparr.tostring()

def string_to_numpy(string:str):
    return np.fromstring( string, dtype=np.int32)
    
def bgp_graph_construction(bgps: list, return_graphs=True, filter=False):
    def is_trp_legal(tp: TriplePattern) -> bool:
        if tp.predicate.pred_freq == -1 or tp.predicate.pred_freq == -1 or tp.predicate.pred_literals == -1 or tp.predicate.pred_subject_count == -1 or tp.predicate.pred_object_count == -1:
            return False
        return True
    
    def is_bpg_legal(bgp:BGP) -> bool:
        for tp in bgp.triples:
            if not is_trp_legal(tp):
                return False
        return True
    
    new_bgps = []
    for x in bgps:
        if filter:
            if is_bpg_legal(x):
                if return_graphs:
                    new_bgps.append(BGPGraph(x))
                else:
                    new_bgps.append(x)
        else:
            if return_graphs:
                new_bgps.append(BGPGraph(x))
            else:
                new_bgps.append(x)
    return new_bgps

# first argument of loading function should be data_path
def load_obj_w_function(path, pickle_file, function, *args, **kwargs):
    bgps  = unpickle_obj(pickle_file)
    if bgps == None:
        bgps = function(path, *args, **kwargs)
        pickle_obj(bgps, pickle_file)
    return bgps

"""def stratified_split(parser:configparser.ConfigParser):
    parser = configparser.ConfigParser()
    parser.read(PATH_TO_CONFIG)
    #bgps = load_BGPS_from_json('/work/data/train_data.json')
    #bgps = filtered_query_missing_entity(parser)
    bgps = filter_missing_query_predicate(bgps,parser)"""
    


def get_node_classes(string):
    match string:
        case "Node": return Node
        case "node_num_pred": return Node_num_pred_encoding
def process_bgp_str(string):
    splits = string.split('{')
    bgp_str = ''
    for i in range(1,len(splits)):
        bgp_str += splits[i].split('}')[0]
        if i > 1:
            bgp_str += ','
    bgp_str = bgp_str.replace('OPTIONAL', '')
    bgp_str = bgp_str.replace(' .',',')
    bgp_str = ' '.join(bgp_str.split())
    return '['+bgp_str.replace('\s', ' ') +']'

def convert_leaf_to_json(input_path, output, time_out=None):
    data = json.load(open(input_path,'r'))
    optional_count = 0
    clause_count = 0
    json_data = {}
    for d in data:
        if isinstance(time_out, int):
            if d['query'] == '':
                continue
            if d['bloom_runtime'] == 'timeout' :
                d['bloom_runtime'] = time_out * 1e15
            if d['jena_runtime'] == 'timeout':
                d['jena_runtime'] = time_out* 1e15
            if d['leapfrog'] == 'timeout':
                d['leapfrog'] = time_out* 1e15
        else:
            if d['query'] == '' or d['bloom_runtime'] == 'timeout' or d['jena_runtime'] == 'timeout' or d['leapfrog'] == 'timeout':
                continue
        if 'OPTIONAL' in d['query']:
            optional_count += 1
        splits = d['query'].split('{')
        if len(splits) != 2:
            clause_count += 1
        dct = {'bloom_runtime': int(d['bloom_runtime']), 'jena_runtime':int(d['jena_runtime']), 'leapfrog':int(d['leapfrog']), 'path':d['path']}
        json_data[process_bgp_str(d['query'])] = dct
        
    print(optional_count,clause_count)
    json.dump(json_data, open(output,'w'))

if __name__ == "__main__":
    
    query = "SELECT (COUNT(*) AS ?count) WHERE { <http://www.wikidata.org/entity/Q520713> ?p ?y . <http://www.wikidata.org/entity/Q520713> <http://www.wikidata.org/prop/direct/P2268> ?z . <http://www.wikidata.org/entity/Q520713> <http://www.wikidata.org/prop/direct-normalized/P269> ?u } LIMIT 10000"
    query = "SELECT (COUNT(*) AS ?count) WHERE { ?x1 <http://www.wikidata.org/prop/direct/P268> ?x2 . ?x1 <http://www.wikidata.org/prop/direct/P166> ?x3 . OPTIONAL { ?x3 <http://www.wikidata.org/prop/direct/P991> ?x4 . ?x3 <http://www.wikidata.org/prop/direct/P585> ?x5 } } LIMIT 10000"
    splits = query.split('{')
    print(process_bgp_str(query))
    input_path = '/work/data/data_files/all_data.json'
    output_path = '/work/data/data_files/converted_all_data.json'
    convert_leaf_to_json(input_path,output_path )
    exit()
    path_to_bgps = '/work/data/bgps.pickle'
    path_predicate_feat_gen = '/work/data/pred_feat.pickle'
    bgps = unpickle_obj(path_to_bgps)
    bins = 30
    pred_topk = 15
    limit_BGPs = None
    Node.pred_feaurizer = PredicateFeaturesQuery.prepare_pred_featues_for_bgp( path_predicate_feat_gen, bins=bins)
    Node.ent_featurizer = None
    Node.pred_bins = bins
    Node.pred_topk = pred_topk
    Node.pred_feat_sub_obj_no = True
    Node.use_ent_feat = False
    if bgps == None:
        bgps = load_BGPS_from_json('/work/data/train_data.json',limit_bgp=limit_BGPs, node=Node)
        pickle_obj(bgps,path_to_bgps)
    bgp_count = len(bgps)
    #pred_featurizer = PredicateFeaturesQuery.prepare_pred_featues_for_bgp(path_predicate_feat_gen, bins=bins)
    bgps = bgp_graph_construction(bgps)
    print(f"Filtered {bgp_count-len(bgps)} of {bgp_count}")   
    
    exit()
    preds, no_unique_preds = get_predicate_with_no_bin()
    print(preds,no_unique_preds)
    
    
    print(f'BGPS loaded : {len(bgps)}')
    single_bgp_path = '/work/data/temp_bgp.pickle'
    
    bgp_g = BGPGraph(bgps[0])
    #pickle_obj(bgp_g,single_bgp_path)
    
    #bgp_g = unpickle_obj(single_bgp_path)
    bgp_g.create_graph()
    print('node representation')
    print(bgp_g.get_node_representation())
    print(bgp_g.get_edge_list())
    print(f'Ground truth is {bgp_g.gt}')
    #ground_truth_distibution(bgps,verbose=True)
    
    
    ents = get_entities(bgps)
    preds = get_predicates(bgps)

    print(f'Entities extracted: {len(ents)}')
    print(f'Preds extracted: {len(preds)}')