from feature_extraction.sparql_query import Query
import json
import pickle as pcl
import time, pandas as pd
import traceback

class PredicateFeaturesQuery(Query):
    def __init__(self, endpoint_url, timeout=30):
        super().__init__(endpoint_url)
        self.start = time.time()
        self.uniqueLiteralCounter = {}
        self.unique_entities_counter = {}
        self.predicate_freq = {}
        self.sparql.setTimeout(timeout)
    #    
    def extract_predicate_features(self, predicates=None, save_decode_err_preds= '/work/data/decode_error_pred.json', save_path=None):
        if predicates == None:
            predicates = self.predicates
        decode_errors = []
        for pred_no,pred in enumerate(predicates):
            try:
                self.set_query_unique_literal_predicate_v2(pred, number=pred_no)
            except json.decoder.JSONDecodeError or Exception:
                #traceback.print_exc()
                decode_errors.append(pred)
            try:
                self.set_query_unique_entity_predicate(pred, number=pred_no)
            except json.decoder.JSONDecodeError or Exception:
                #traceback.print_exc()
                decode_errors.append(pred)
            try:
                self.set_predicate_freq(pred, number=pred_no)
            except json.decoder.JSONDecodeError or Exception:
                #traceback.print_exc()
                decode_errors.append(pred)
            if save_path != None and (pred_no % 1000 == 1):
                self.save(save_path)
        json.dump(predicates,open(save_decode_err_preds,'w'))
    #
    def set_query_unique_literal_predicate(self, predicate, number=None):
        query_str = f'''SELECT ?s ?o WHERE {{
            ?s <{predicate}> ?o .
        }}
        '''
        res = self.run_query(query_str)
        self.process_freq_features(res, predicate, number=number)
    #
    def set_query_unique_literal_predicate_v2(self, predicate, number=None):
        query_str = f'''SELECT (COUNT(DISTINCT ?o) AS ?literals) WHERE {{
            ?s <{predicate}> ?o .
            FILTER(isLiteral(?o))
        }}
        '''
        try:
            res = self.run_query(query_str)
            self.uniqueLiteralCounter[predicate] = res['results']['bindings'][0]['literals']['value']
            print(f"{time.time()-self.start:.2f}{number}{predicate}: {res['results']['bindings'][0]['literals']['value']}")
        except RuntimeError or Exception:
            self.uniqueLiteralCounter[predicate] = -1
        # self.process_freq_features(res, predicate, number=number)
    #
    def set_query_unique_entity_predicate(self, predicate, number=None):
        query_str = f'''SELECT (COUNT(DISTINCT ?e) AS ?entities) WHERE {{
            {{?e <{predicate}> ?o .
            FILTER(isURI(?e))}}
            UNION {{?s <{predicate}> ?e .
            FILTER(isURI(?e))}}
        }}
        '''
        try:
            res = self.run_query(query_str)
            print(f"ENT {predicate}: {res['results']['bindings'][0]['entities']['value']}")
            self.unique_entities_counter[predicate] = res['results']['bindings'][0]['entities']['value']
            
        except RuntimeError or Exception as e:
            print(e.pri)
            self.unique_entities_counter[predicate] = -1
    #
    def set_predicate_freq(self, predicate, number=None):
        query_str = f'''SELECT (COUNT(*) AS ?triples) WHERE {{
            ?s <{predicate}> ?o .
        }}
        '''
        try:
            res = self.run_query(query_str)
            self.predicate_freq[predicate] = res['results']['bindings'][0]['triples']['value']
            print(f"FREQ {predicate}: {res['results']['bindings'][0]['triples']['value']}")
        except RuntimeError or Exception:
            self.predicate_freq[predicate] = -1
    #   
    def process_freq_features(self, sparql_result, predicate, verbose=True, number = None):
        unique_literals = set()
        unique_ents = set()
        pattern_count = 0
        print(f"Total binding for {predicate}: {len(sparql_result['results']['bindings'])}")
        for x in sparql_result['results']['bindings']:
            pattern_count += 1
            if x['o']['type'] == 'literal':
                unique_literals.add(x['o']['value']) 
            elif x['o']['type'] == 'uri':
                unique_ents.add(x['o']['value'])
            if x['s']['type'] == 'uri':
                unique_ents.add(x['s']['value'])
        self.uniqueLiteralCounter[predicate] = len(unique_literals)
        self.unique_entities_counter[predicate] = len(unique_ents)
        self.predicate_freq[predicate] = pattern_count
        if verbose:
            self.print_predicat_stat(predicate,len(unique_literals),len(unique_ents), pattern_count, number=number)
    #
    def save(self, path):
        with open(path,'wb') as f:
            pcl.dump(self, f) 
    #
    def load(self, path):
        with open(path,'rb') as f:
            self = pcl.load(f) 
    #
    def print_predicat_stat(self,predicate : str, unique_literals : int, \
                            unique_ents : int, pattern_count : int, number=None):
        if number == None:
            number = ''
        else:
            number = f' : {number:05}'
        print(f"{time.time()-self.start:.2f}{number} Stats for {predicate}")
        print(f"\t# of triples: {pattern_count}")
        print(f"\t# of unique entities: {unique_ents}")
        print(f"\t# of unique literals: {unique_literals}")
    #
    def get_rdf_predicates(self, save_path=None) -> list:
        query_str = f''' SELECT DISTINCT ?p WHERE {{
            ?s ?p ?o
        }}
        '''
        predicates = []
        res = self.run_query(query_str)
        for x in res['results']['bindings']:
            predicates.append(x['p']['value'])
        if save_path != None:
            with open(save_path,'w') as f:
                json.dump(predicates,f)
        return predicates
    #
    def load_predicates(self,path):
        self.predicates = json.loads(open(path,'r').read())
        return self.predicates
    #
    def predicate_binner(self, bins = 30):
        dct = {'predicate':[], 'freq':[]}
        for k in self.predicate_freq.keys():
            dct['predicate'].append(k)
            dct['freq'].append(self.predicate_freq[k])
        df = pd.DataFrame.from_dict(dct)
        df['bin'], cut_bin = pd.qcut(df['freq'], q = bins, labels = [x for x in range(bins)], retbins = True)
        df = df.set_index('predicate')
        self.predicate_binner = df
        self.bin_vals = cut_bin
        self.total_bin = bins
    #
    def get_bin(self, predicate):
        return self.predicate_binner.loc[predicate]['bin']
    #
    #deprecated
    def binnify(self,predicate_freq):
        bin_counter = 0
        for x in range(1, len(self.bin_vals)):
            if self.bin_vals[x-1] < predicate_freq and predicate_freq < self.bin_vals[x]:
                return bin_counter
            bin_counter += 1
        return bin_counter
        
        
def iterate_results(sparql_results):
    for x in sparql_results['results']['bindings']:
        for v in sparql_results['head']['vars']:
            print(v,x[v]) 
            
def print_bindings_stats(sparql_results):
    binding_count = 0
    literals = []
    for x in sparql_results['results']['bindings']:
        binding_count +=1
    print(f"number of bindings are {binding_count}")

def test():
    q = Query("http://dbpedia.org/sparql")
    q.run_query("SELECT * WHERE { ?s ?p ?o} LIMIT 1")
    print(q.results.keys())
    iterate_results(q.results)
    #for x in q.results['results']['bindings']:
    #    for v in q.results['head']['vars']:
    #        print(v,x[v])

def testUniqueLiteralQuery():
    q = PredicateFeaturesQuery("https://query.wikidata.org/sparql")
    #q = PredicateFeaturesQuery("http://172.21.232.208:3030/jena/sparql")
    #preds = q.get_rdf_predicates(save_path='/work/data/predicates.json')
    preds = q.load_predicates('/work/data/specific_graph/predicates.json')
    print(f"# preds {len(preds)}")
    q.extract_predicate_features(preds, save_path='/work/data/specific_graph/pred_feat.pickle')
    q.save('/work/data/pred_feat.pickle')
    #q.set_query_unique_literal_predicate("<http://www.wikidata.org/prop/direct/P5395>")
    #iterate_results(q.results)
    #print_bindings_stats(q.results)
    
if __name__ == "__main__":
    #test()
    testUniqueLiteralQuery()
    