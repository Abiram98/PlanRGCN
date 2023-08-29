from feature_extraction.predicates.predicate_features import PredicateFeaturesQuery
from graph_construction.nodes.node import Node
from graph_construction.triple_pattern import TriplePattern
from feature_extraction.entity_features import EntityFeatures

class BGP:
    def __init__(self, BGP_string:str, info:dict, node_class = Node, TP_class= TriplePattern):
        '''BGP_string, info dict, node_class: node type (Node is default), TP_class: Triple pattern representation class (Default is TriplePattern). 
        Ground truth value is assigned based on info['gt'] boolean'''
        self.bgp_string = BGP_string
        self.node_class = node_class
        triple_strings = BGP_string[1:-1].split(',')
        self.triples = []
        for t in triple_strings:
            if len(t) == 0:
                continue
            temp_split = t.split(' ')
            temp_split = [s for s in temp_split if s != '']
            if len(temp_split) == 0:
                continue
            self.triples.append(TP_class(t, node_class=node_class))
        #TODO check whether this does something:
        #if predicate_stat != None:
        #    self.total_bins = predicate_stat.total_bin
        
        self.data_dict = info
        #self.ground_truth = 1 if info['with_runtime']<info['without_runtime'] else 0
        #self.ground_truth = 1 if info['leaffrog_runtime']<info['jena_runtime'] else 0
        #self.ground_truth = float(info['with_runtime'])
        #self.ground_truth = float(info['without_size'])
        #self.ground_truth = 1 if len(self.triples) < 4 else 0
        try:
            self.ground_truth = 1 if info['gt'] else 0
        except:
            print(info)
        #self.ground_truth = 1 if 0.01 < (info['without_runtime'] * 10e-9) else 0
    
    #def set_predicate_feat_gen(self,predicate_stat: PredicateFeaturesQuery):
    #    self.predicate_stat = predicate_stat
    
    def __str__(self):
        temp_str = 'BGP( '
        for t in self.triples:
            temp_str = temp_str +' '+ str(t)
        temp_str = temp_str +' )'
        return temp_str