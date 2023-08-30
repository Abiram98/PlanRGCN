import json
import os
import argparse
from feature_extraction.sparql import Endpoint

class ExtractorBase:
    def __init__(self, endpoint:Endpoint, output_dir:str, output_file='predicates.json') -> None:
        self.endpoint = endpoint
        self.output_dir = output_dir
        self.output_file = output_file
        
class PredicateExtractor(ExtractorBase):
    def __init__(self, endpoint: Endpoint, output_dir: str, output_file='predicates.json') -> None:
        super().__init__(endpoint, output_dir, output_file)
    
    def get_predicates_query(self):
        query = """SELECT DISTINCT ?p WHERE {
            ?s ?p ?o
        }
        """
        res = self.endpoint.run_query(query)
        res_fp = f"{self.output_dir}/predicate_result.json"
        json.dump(res, open(res_fp, 'w'))
        print('Predicates extracted')
        predicates = []
        for bind in res['results']['bindings']:
            predicates.append(bind['p']['value'])
        return predicates
    
    def save_predicates(self):
        preds = self.get_predicates_query()
        json.dump(preds, open(f"{self.output_dir}/{self.output_file}", 'w'))
        print('Predicated Saved')
        
class PredicateFreqExtractor(ExtractorBase):
    def __init__(self, endpoint: Endpoint, output_dir: str, output_file='predicates.json') -> None:
        super().__init__(endpoint, output_dir, output_file)
        self.predicates = json.load(open(f"{self.output_dir}/{self.output_file}", 'r'))
    
    def group_predicates(self):
        
        pass
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Predicate Util',
                    description='Utility for predicate feature extraction',
                    epilog='Text at the bottom of help')
    
    parser.add_argument('task')
    parser.add_argument('-e', '--endpoint')
    parser.add_argument('--dir', '--output_dir')
    parser.add_argument('--pred_file')
    
    args = parser.parse_args()
    
    if args.task == 'extract-predicates':
        output_dir = f"{args.dir}/predicate"
        os.system(f"mkdir -p {output_dir}")
        endpoint = Endpoint(args.endpoint)
        pred_ex = PredicateExtractor(endpoint, output_dir, output_file=args.pred_file)
        pred_ex.save_predicates()
    elif args.task == 'extract-co-predicates':
        output_dir = f"{args.dir}/predicate"
        os.system(f"mkdir -p {output_dir}")
        
        pass
        
    
        