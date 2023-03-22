from sparql_query import Query
import json
import pickle as pcl
import time

class PredicateFeaturesQuery(Query):
    def __init__(self, endpoint_url):
        super().__init__(endpoint_url)
        self.start = time.time()
        self.uniqueLiteralCounter = {}
        self.unique_entities_counter = {}
        self.predicate_freq = {}
        
    def extract_predicate_features(self, predicates=None, save_decode_err_preds= '/work/data/decode_error_pred.json'):
        if predicates == None:
            predicates = self.predicates
        decode_errors = []
        for pred_no,pred in enumerate(predicates):
            try:
                self.set_query_unique_literal_predicate(pred, number=pred_no)
            except json.decoder.JSONDecodeError:
                decode_errors.append(pred)
        json.dump(predicates,open(save_decode_err_preds,'w'))
    
    def set_query_unique_literal_predicate(self, predicate, number=None):
        query_str = f'''SELECT ?s ?o WHERE {{
            ?s <{predicate}> ?o .
        }}
        '''
        res = self.run_query(query_str)
        self.process_freq_features(res, predicate, number=number)
        
    def process_freq_features(self, sparql_result, predicate, verbose=True, number = None):
        unique_literals = set()
        unique_ents = set()
        pattern_count = 0
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
    
    def save(self, path):
        with open(path,'wb') as f:
            pcl.dump(self, f) 
    
    def load(self, path):
        with open(path,'rb') as f:
            self = pcl.load(f) 
    
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
    
    def load_predicates(self,path):
        self.predicates = json.loads(open(path,'r').read())
        return self.predicates
        
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
    #preds = q.get_rdf_predicates(save_path='/work/data/predicates.json')
    preds = q.load_predicates('/work/data/predicates.json')
    print(f"# preds {len(preds)}")
    q.extract_predicate_features(preds)
    q.save('/work/data/pred_feat.pickle')
    #q.set_query_unique_literal_predicate("<http://www.wikidata.org/prop/direct/P5395>")
    #iterate_results(q.results)
    #print_bindings_stats(q.results)
    
if __name__ == "__main__":
    #test()
    testUniqueLiteralQuery()
    