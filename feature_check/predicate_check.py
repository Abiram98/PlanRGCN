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
        q.extract_predicate_features(predicates=predicates,save_decode_err_preds=decode_error_path, save_path=save_path)
    q.sparql = SPARQLWrapper(endpoint_url)
    q.sparql.setReturnFormat(JSON)
    q.sparql.setMethod(POST)
    error_preds = json.load(open(parser['PredicateFeaturizerSubObj']['load_error_path']))
    p = parser['PredicateFeaturizerSubObj']['save_path']
    save_path = f'{p}{datetime.now().strftime("%d_%m_%Y_%H_%M")}.pickle'
    p = parser['PredicateFeaturizerSubObj']['error_path']
    decode_error_path = f'{p}{datetime.now().strftime("%d_%m_%Y_%H_%M")}.json'
    #q.extract_features_for_remaining(predicates=error_preds, save_decode_err_preds= decode_error_path, save_path=save_path)
    
    no_pred_freq,no_pred_literal, no_pred_ent= missing_predicate_check(q, error_preds)
    
    missing_pred = check_missing_query_preds(parser,q)
    print(missing_pred)
    print(f"Prediates: Missing frequency info for {len(no_pred_freq)}, Missing entity info for {len(no_pred_ent)}, Missing literals info for {len(no_pred_literal)}")
    print(f"Missing preds from queries : {len(missing_pred)}")
    

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
        
def missing_predicate_check(q, error_preds):
    no_pred_freq = []
    no_pred_literal = []
    no_pred_ent = []
    for e in error_preds:
        try:
            q.predicate_freq[e]
        except KeyError:
            no_pred_freq.append(e)
        try:
            q.unique_entities_counter[e]
        except KeyError:
            no_pred_ent.append(e)
        try:
            q.uniqueLiteralCounter[e]
        except KeyError:
            no_pred_literal.append(e)
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