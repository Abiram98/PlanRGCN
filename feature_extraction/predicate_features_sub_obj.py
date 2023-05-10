from feature_extraction.predicates.predicate_features import PredicateFeaturesQuery, check_stats,load_pickle
import json
from datetime import datetime
import configparser
from feature_extraction.constants import PATH_TO_CONFIG
import os
import argparse
from SPARQLWrapper import SPARQLWrapper,JSON,POST

class Predicate_Featurizer_Sub_Obj(PredicateFeaturesQuery):
    def __init__(self, endpoint_url=None, timeout=30):
        super().__init__(endpoint_url, timeout)
    
    def initialise_sparql(self,endpoint_url):
        self.sparql = SPARQLWrapper(endpoint_url)
        self.sparql.setReturnFormat(JSON)
        self.sparql.setMethod(POST)
    
    def set_query_unique_entity_predicate(self, predicate, number=None):
        subject_count, object_count = -1,-1
        query_str = f'''
        SELECT (COUNT(DISTINCT ?e) AS ?subjects) WHERE {{
            ?e <{predicate}> ?o .
        }}
        '''
        try:
            res = self.run_query(query_str)
            subject_count = res['results']['bindings'][0]['subjects']['value']
        except RuntimeError or Exception or TimeoutError as e:
            return None

        
        query_str = f'''
        SELECT (COUNT(DISTINCT ?o) AS ?objects) WHERE {{
            ?e <{predicate}> ?o .
            FILTER(isURI(?o))
        }}
        '''
        try:
            res = self.run_query(query_str)
            
            object_count = res['results']['bindings'][0]['objects']['value']
        except RuntimeError or Exception or TimeoutError as e:
            return None
            pass
        print(f"ENT {predicate}: (SUB) {subject_count} (OBJ) {object_count}")
        self.unique_entities_counter[predicate] = (subject_count,object_count)
        return (subject_count,object_count)
    
    def convert_dict_vals_to_int(self, dct:dict):
        for k in dct.keys():
            val = dct[k]
            if isinstance(val, tuple):
                val = (int(val[0]),int(val[1]) )
            else:
                val = int(val)
            dct[k] = val
        return dct
    
    #Used to load existing object
    def load(path, obj_type= None):
        obj = load_pickle(path)
        if hasattr(obj,'endpoint_url'):
            endpoint_url = obj.endpoint_url
        else:
            endpoint_url = None
        if obj_type != None:
            i = obj_type(endpoint_url)
        else:
            i = Predicate_Featurizer_Sub_Obj(endpoint_url)
        i.uniqueLiteralCounter = i.convert_dict_vals_to_int( obj.uniqueLiteralCounter)
        i.predicate_freq = i.convert_dict_vals_to_int(obj.predicate_freq)
        i.unique_entities_counter = i.convert_dict_vals_to_int(obj.unique_entities_counter)
        return i
    
    def prepare_pred_featues_for_bgp(path, bins = 30, topk =15, obj_type=None):
        i = Predicate_Featurizer_Sub_Obj.load(path, obj_type=obj_type)
        i.prepare_pred_feat(bins=bins,k=topk)
        return i  

#with virtuoso endpoint 33.93 s
def run_featurizer(parser:configparser.ConfigParser):
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
        predicates = q.get_rdf_predicates(save_path=parser['PredicateFeaturizerSubObj']['predicate_path'])
    
    #predicates = json.load(open('/work/data/confs/newPredExtractionRun/predicates_only.json','r'))
    print(f"Number of total predicates: {len(predicates)}")
    #path_featurizer ='/work/data/confs/newPredExtractionRun/pred_feat_w_sub_obj.pickle'
    
    q.extract_predicate_features(predicates=predicates,save_decode_err_preds=decode_error_path, save_path=save_path)
    
if __name__ == "__main__":
    arg_parse = argparse.ArgumentParser(prog='PredicateFeaturizer_subj_obj)')
    arg_parse.add_argument('cmd')
    args = arg_parse.parse_args()
    
    parser = configparser.ConfigParser()
    parser.read(PATH_TO_CONFIG)
    match( args.cmd): 
        case 'run':
            run_featurizer(parser)
        case 'plot':
            pass
        case other:
            print('Please choose a valid option')
    #Prepare predicate Featurizer