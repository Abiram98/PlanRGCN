from feature_extraction.predicate_features import PredicateFeaturesQuery, check_stats
import json, SPARQLWrapper
from datetime import datetime

class Predicate_Featurizer_Sub_Obj(PredicateFeaturesQuery):
    def __init__(self, endpoint_url=None, timeout=30):
        super().__init__(endpoint_url, timeout)
    
    def set_query_unique_entity_predicate(self, predicate, number=None):
        subject_count, object_count = -1,-1
        query_str = f'''
        SELECT (COUNT(DISTINCT ?e) AS ?subjects) WHERE {{
            ?e <{predicate}> ?o .
            FILTER(isURI(?e))
        }}
        '''
        try:
            res = self.run_query(query_str)
            subject_count = res['results']['bindings'][0]['subjects']['value']
        except RuntimeError or Exception as e:
            pass
        
        query_str = f'''
        SELECT (COUNT(DISTINCT ?o) AS ?objects) WHERE {{
            ?e <{predicate}> ?o .
            FILTER(isURI(?o))
        }}
        '''
        try:
            res = self.run_query(query_str)
            
            object_count = res['results']['bindings'][0]['objects']['value']
        except RuntimeError or Exception as e:
            pass
        print(f"ENT {predicate}: (SUB) {subject_count} (OBJ) {object_count}")
        self.unique_entities_counter[predicate] = (subject_count,object_count)

#with virtuoso endpoint 33.93 s
def run_featurizer():
    predicates = json.load(open('/work/data/confs/newPredExtractionRun/predicates_only.json','r'))
    print(f"Number of total predicates: {len(predicates)}")
    path_featurizer ='/work/data/confs/newPredExtractionRun/pred_feat_w_sub_obj.pickle'
    endpoint_url = 'http://172.21.233.23:8891/sparql/'
    save_path = f'/work/data/confs/newPredExtractionRun/pred_feat__w_sub_obj{datetime.now().strftime("%d_%m_%Y_%H_%M")}.pickle'
    decode_error_path = f'/work/data/confs/newPredExtractionRun/decode_error_pred_sub_obj{datetime.now().strftime("%d_%m_%Y_%H_%M")}.json'
    q = Predicate_Featurizer_Sub_Obj(endpoint_url=endpoint_url,timeout=None)
    q.extract_predicate_features(predicates=predicates,save_decode_err_preds=decode_error_path, save_path=save_path)
    
if __name__ == "__main__":
    run_featurizer()
    #Prepare predicate Featurizer