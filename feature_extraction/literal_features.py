from feature_extraction.sparql_query import Query
import json, os, configparser
from feature_extraction.constants import PATH_TO_CONFIG

class LiteralFeat(Query):
    def __init__(self, endpoint_url=None):
        super().__init__(endpoint_url)
        self.literal_type = {}
    
    def get_rdf_literals(self, save_path=None, save_literal_types=None) -> list:
        query_str = f''' SELECT DISTINCT ?o (datatype(?o) as ?datatype) WHERE {{
             ?s ?p ?o .
            FILTER isLiteral(?o)
        }}
        '''
        literals = []
        res = self.run_query(query_str)
        for i in range(len(res['results']['bindings'])):
            val = res['results']['bindings'][i]['o']['value']
            if 'datatype' in res['results']['bindings'][i].keys():
                t = res['results']['bindings'][i]['datatype']['value']
            else:
                t = 'untyped'
            literals.append(val)
            self.literal_type[val] = t
            pass
        #for x in res['results']['bindings']:
        #    literals.append(x['o']['value'])
        #    self.literal_type[x['o']['value']] = x
        if save_path != None:
            with open(save_path,'w') as f:
                json.dump(literals,f)
        if save_literal_types != None:
            with open(save_literal_types,'w') as f:
                json.dump(self.literal_type, f)
        return literals
    
    def get_literal_types(self):
        types = set()
        for x in self.literal_type.values():
            types.add(x)
        return types
    
    def get_str_literal_types(self, path_to_save=None):
        language_tags = {}
        query_str = '''
        SELECT DISTINCT ?o (lang(?o) as ?tag) WHERE {{
             ?s ?p ?o .
            
            FILTER (lang(?o) != "")
        }}
        '''
        res = self.run_query(query_str)
        for i in range(len(res['results']['bindings'])):
            val = res['results']['bindings'][i]['o']['value']
            t = res['results']['bindings'][i]['tag']['value']
            language_tags[val] = t
        self.language_tags = language_tags
        
        if path_to_save != None:
            with open(path_to_save,'w') as f:
                json.dump(language_tags, f)
        return language_tags

literal_featurizer = None
def literal_stat_extraction(parser:configparser.ConfigParser):
    global literal_featurizer, literals,language_tags
    endpoint_url = 'http://172.21.233.23:8891/sparql/'
    literal_featurizer = LiteralFeat(endpoint_url)
    literal_path = '/work/data/specific_graph/literals.json'
    literal_type_dict_path = '/work/data/specific_graph/literal_dict.json'
    language_tag_path = '/work/data/specific_graph/lang_tag_dict.json'
    if not os.path.isfile(literal_path) or not os.path.isfile(literal_type_dict_path)or not os.path.isfile(language_tag_path):
        literals = literal_featurizer.get_rdf_literals(save_path=literal_path, save_literal_types=literal_type_dict_path)
        language_tags = literal_featurizer.get_str_literal_types(path_to_save=language_tag_path)
    else:
        literals = json.load(open(literal_path,'r'))
        literal_featurizer = LiteralFeat(endpoint_url)
        literal_featurizer.literal_type = json.load(open(literal_type_dict_path,'r'))
        literal_featurizer.language_tags = json.load(open(language_tag_path,'r'))
        
    print(f'Number of literals: {len(literals)}')
    
    
def find_duplicated_elements(lst:list):
    non_duplicates = set()
    duplicates = list()
    for e in lst:
        if e in non_duplicates:
            duplicates.append(e)
        else:
            non_duplicates.add(e)
    return non_duplicates,duplicates
def non_int_in_lst(lst:list):
    temp = list()
    for x in lst:
        try:
            int(x)
        except ValueError:
            temp.append(x)
    return temp
def dubplicate_literal_checks():
    literal_stat_extraction()
    global duplicate_lits
    _,duplicate_lits = find_duplicated_elements(literals)
    nonint = non_int_in_lst(duplicate_lits)
    #s = set()
    #for x in duplicate_lits:
    #    s.add(literal_featurizer.literal_type[x])
    #[print(x) for x in s]


if __name__ == "__main__":
    parser = configparser.ConfigParser()
    parser.read(PATH_TO_CONFIG)
    literal_stat_extraction()
    literal_featurizer:LiteralFeat
    print(literal_featurizer.get_literal_types())
    