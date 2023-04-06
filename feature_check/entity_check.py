from feature_extraction.entity_features import EntityFeatures, entity_stat_extraction
import configparser
from feature_extraction.constants import PATH_TO_CONFIG
import os
import json
from graph_construction.bgp import BGP
from graph_construction.triple_pattern import TriplePattern
from utils import load_BGPS_from_json

def checker(config_path = PATH_TO_CONFIG):
    global entities, ent_freq
    parser = configparser.ConfigParser()
    parser.read(config_path)
    if not (os.path.isfile(parser['EntityFeaturizer']['entity_path'])) and not (os.path.isfile(parser['EntityFeaturizer']['entity_freq_dict_path'])):
        entity_stat_extraction(parser)
    entities = json.load(open(parser['EntityFeaturizer']['entity_path']))
    ent_freq = json.load(open(parser['EntityFeaturizer']['entity_freq_dict_path']))
    featurizer = EntityFeatures(parser['endpoint']['endpoint_url'])
    featurizer.freq = ent_freq
    miss_ent = set()
    for e in entities:
        try:
            ent_freq[e]
        except KeyError:
            miss_ent.add(e)
    print(f"Missing entity features for {len(list(miss_ent))}")
    missing_ent_query = check_missing_query_entities(parser, featurizer)
    print(f"Missing entities in RDF graph from queries: {len(missing_ent_query)}")
    if 'missing_query_path' in parser['EntityFeaturizer'].keys():
        json.dump(list(missing_ent_query), open(parser['EntityFeaturizer']['missing_query_path'],'w'))
    
def check_missing_query_entities(parser:configparser.ConfigParser, q: EntityFeatures):
    bgps = load_BGPS_from_json(parser['Dataset']['train_data_path'])
    print(f"amount fo bgps : {len(bgps)}")
    missing_ent_query = set()
    for x in bgps:
        x: BGP
        for t in x.triples:
            t:TriplePattern
            try:
                if t.object.type == 'URI':
                    q.freq[t.object.node_label]
            except KeyError:
                missing_ent_query.add(t.object.node_label)
            try:
                if t.subject.type == 'URI':
                    q.freq[t.subject.node_label]
            except KeyError:
                missing_ent_query.add(t.subject.node_label)
    return missing_ent_query 

def filtered_query_missing_entity(parser:configparser.ConfigParser):
    ent_freq = json.load(open(parser['EntityFeaturizer']['entity_freq_dict_path']))
    featurizer = EntityFeatures(parser['endpoint']['endpoint_url'])
    featurizer.freq = ent_freq
    bgps = load_BGPS_from_json(parser['Dataset']['train_data_path'])
    filtered_bgps = []
    for x in bgps:
        x: BGP
        try:
            for t in x.triples:
                t:TriplePattern
                if t.object.type == 'URI':
                    featurizer.freq[t.object.node_label]
                if t.subject.type == 'URI':
                    featurizer.freq[t.subject.node_label]
            filtered_bgps.append(x)
        except KeyError:
            ...
    return filtered_bgps

if __name__ == "__main__":
    checker()