from feature_extraction.predicate_features_sub_obj import Predicate_Featurizer_Sub_Obj
import configparser
from feature_extraction.constants import PATH_TO_CONFIG
from datetime import datetime
import json, os
from SPARQLWrapper import SPARQLWrapper,JSON,POST
from graph_construction.bgp import BGP
from graph_construction.triple_pattern import TriplePattern
from utils import load_BGPS_from_json

def checker(config_path = PATH_TO_CONFIG):
    global q,no_pred_freq,no_pred_literal,no_pred_ent,missing_pred
    parser = configparser.ConfigParser()
    parser.read(PATH_TO_CONFIG)
    endpoint_url = parser['endpoint']['endpoint_url']
    #save_path = f'/work/data/confs/newPredExtractionRun/pred_feat__w_sub_obj{datetime.now().strftime("%d_%m_%Y_%H_%M")}.pickle'
    p = parser['PredicateFeaturizerSubObj']['save_path']
    save_path = f'{p}{datetime.now().strftime("%d_%m_%Y_%H_%M")}.pickle'
    p = parser['PredicateFeaturizerSubObj']['error_path']
    decode_error_path = f'{p}{datetime.now().strftime("%d_%m_%Y_%H_%M")}.json'
    q = Predicate_Featurizer_Sub_Obj(endpoint_url=endpoint_url,timeout=None)
    
    if os.path.isfile(parser['PredicateFeaturizerSubObj']['predicate_path']):
        predicates = json.load(open(parser['PredicateFeaturizerSubObj']['predicate_path'],'r'))
    else:
        print("Extracting Predicates")
        predicates = q.get_rdf_predicates(save_path=parser['PredicateFeaturizerSubObj']['predicate_path'])
    q.predicates = predicates
    #predicates = json.load(open('/work/data/confs/newPredExtractionRun/predicates_only.json','r'))
    print(f"Number of total predicates: {len(predicates)}")
    #path_featurizer ='/work/data/confs/newPredExtractionRun/pred_feat_w_sub_obj.pickle'
    if os.path.isfile(parser['PredicateFeaturizerSubObj']['load_path']):
        q = Predicate_Featurizer_Sub_Obj.load(parser['PredicateFeaturizerSubObj']['load_path'])
    else:
        q.sparql = SPARQLWrapper(endpoint_url)
        q.sparql.setReturnFormat(JSON)
        q.sparql.setMethod(POST)
        q.extract_predicate_features(predicates=predicates,save_decode_err_preds=decode_error_path, save_path=save_path)
    
    error_preds = json.load(open(parser['PredicateFeaturizerSubObj']['load_error_path']))
    p = parser['PredicateFeaturizerSubObj']['save_path']
    save_path = f'{p}{datetime.now().strftime("%d_%m_%Y_%H_%M")}.pickle'
    p = parser['PredicateFeaturizerSubObj']['error_path']
    decode_error_path = f'{p}{datetime.now().strftime("%d_%m_%Y_%H_%M")}.json'
    #q.extract_features_for_remaining(predicates=error_preds, save_decode_err_preds= decode_error_path, save_path=save_path)
    no_pred_freq,no_pred_literal, no_pred_ent= missing_predicate_check(q, predicates)
    print(f"Prediates: Missing frequency info for {len(no_pred_freq)}, Missing entity info for {len(no_pred_ent)}, Missing literals info for {len(no_pred_literal)}")
    no_pred_freq,no_pred_literal, no_pred_ent= minus1_predicate_check(q, predicates)
    print(f"Prediates: (-1) frequency info for {len(no_pred_freq)}, (-1) entity info for {len(no_pred_ent)}, (-1) literals info for {len(no_pred_literal)}")
     
    missing_pred = check_missing_query_preds(parser,q)
    print(f"Missing preds from queries : {len(missing_pred)}")
    find_bgp_to_filter(parser,q)
    

def check_missing_query_preds(parser:configparser.ConfigParser, q: Predicate_Featurizer_Sub_Obj):
    bgps = load_BGPS_from_json(parser['Dataset']['train_data_path'])
    print(f"amoutn fo bgps : {len(bgps)}")
    missing_pred = set()
    for x in bgps:
        x: BGP
        for t in x.triples:
            t:TriplePattern
            try:
                q.predicate_freq[t.predicate.node_label]
            except KeyError:
                missing_pred.add(t.predicate.node_label)
    return missing_pred

def find_bgp_to_filter(parser:configparser.ConfigParser, q: Predicate_Featurizer_Sub_Obj):
    bgps = load_BGPS_from_json(parser['Dataset']['train_data_path'])
    print(f"amount of bgps : {len(bgps)}")
    filter_count = 0
    for x in bgps:
        x: BGP
        try:
            for t in x.triples:
                t:TriplePattern
                q.predicate_freq[t.predicate.node_label]
        except KeyError:
            filter_count +=1
    print(f"BGPs without predicate features: {filter_count}")
    return filter_count

def missing_predicate_check(q, error_preds):
    no_pred_freq = set()
    no_pred_literal = set()
    no_pred_ent = set()
    for e in error_preds:
        try:
            q.predicate_freq[e]
        except KeyError:
            no_pred_freq.add(e)
        try:
            q.unique_entities_counter[e]
        except KeyError:
            no_pred_ent.add(e)
        try:
            q.uniqueLiteralCounter[e]
        except KeyError:
            no_pred_literal.add(e)
    return no_pred_freq,no_pred_literal, no_pred_ent
def minus1_predicate_check(q, error_preds):
    no_pred_freq = set()
    no_pred_literal = set()
    no_pred_ent = set()
    for e in error_preds:
        try:
            if q.predicate_freq[e] == -1:
                no_pred_freq.add(e)
        except KeyError:
            pass
        try:
            if q.unique_entities_counter[e] == -1:
                no_pred_ent.add(e)
        except KeyError:
            pass
        try:
            if q.uniqueLiteralCounter[e] == -1:
                no_pred_literal.add(e)
        except KeyError:
            pass
    return no_pred_freq,no_pred_literal, no_pred_ent


def filter_missing_query_predicate(bgps:list[BGP], parser):
    parser = configparser.ConfigParser()
    parser.read(PATH_TO_CONFIG)
    feat_generation_path = parser['PredicateFeaturizerSubObj']['load_path']
    topk = int(parser['PredicateFeaturizerSubObj']['topk'])
    bin_no = int(parser['PredicateFeaturizerSubObj']['bin_no'])
    pred_feature_rizer = Predicate_Featurizer_Sub_Obj.prepare_pred_featues_for_bgp(feat_generation_path, bins=bin_no, topk=topk)
    
    filtered_bgps = []
    
    for x in bgps:
        x: BGP
        try:
            for t in x.triples:
                t:TriplePattern
                pred_feature_rizer.predicate_freq[t.predicate.node_label]
            filtered_bgps.append(x)
        except KeyError:
            ...
    return filtered_bgps
    
if __name__ == "__main__":
    #Generates all missing features
    checker() #'/work/data/specific_graph/conf.ini'