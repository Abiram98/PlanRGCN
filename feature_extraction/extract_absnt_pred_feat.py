
import json

import SPARQLWrapper
from feature_extraction.predicate_features_sub_obj import Predicate_Featurizer_Sub_Obj



def extract_absent_sub_obj(extracted_dir='/work/data/extracted_statistics'):
    pred_featurizer = Predicate_Featurizer_Sub_Obj(endpoint_url="http://172.21.233.23:83/wikidata/sparql", timeout=None)
    
    all_preds = json.load(open(f"{extracted_dir}/predicates_only.json"))
    ents = json.load(open(f"{extracted_dir}/pred_unique_subj_obj.json"))
    for i in all_preds:
        if i in ents.keys():
            continue
        if i in ['http://www.w3.org/1999/02/22-rdf-syntax-ns#type']:
            continue
        try:
            val = pred_featurizer.set_query_unique_entity_predicate(i)
        except SPARQLWrapper.SPARQLExceptions.EndPointInternalError:
            print(f"ERROR: (Endpoint Heap Size) Did not work for {i} [SubObj]")
            continue
        if val == None:
            print(f"ERROR: Did not work for {i} [SubObj]")
        else:
            ents[i] = val
    json.dump(ents,open(f"{extracted_dir}/updated_pred_unique_subj_obj.json", 'w'))
    
    pred_freq = json.load(open(f"{extracted_dir}/pred_freq.json"))
    for i in all_preds:
            if i in pred_freq.keys():
                continue
            
            try:
                val = pred_featurizer.set_predicate_freq(i)
            except SPARQLWrapper.SPARQLExceptions.EndPointInternalError:
                print(f"ERROR: (Endpoint Heap Size) Did not work for {i} [FREQ]")
                continue
            if val == None:
                print(f"ERROR: Did not work for {i} [FREQ]")
            else:
                pred_freq[i] = val
    json.dump(pred_freq, open(f"{extracted_dir}/updated_pred_freq.json", 'w'))
    

    lits = json.load(open(f"{extracted_dir}/pred_unique_lits.json"))
    for i in all_preds:
            if i in lits.keys():
                continue
            try:
                val = pred_featurizer.set_query_unique_literal_predicate_v2(i)
            except SPARQLWrapper.SPARQLExceptions.EndPointInternalError:
                print(f"ERROR: (Endpoint Heap Size) Did not work for {i} [Literals]")
                continue
            if val == None:
                print(f"ERROR: Did not work for {i} [Literals]")
            else:
                lits[i] = val
    json.dump(lits, open(f"{extracted_dir}/updated_pred_unique_lits.json", 'w'))

if __name__ == "__main__":
    extract_absent_sub_obj()