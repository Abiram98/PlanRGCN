from feature_extraction.predicate_features import PredicateFeaturesQuery
from graph_construction.triple_pattern import TriplePattern
from feature_extraction.entity_features import EntityFeatures
class BGP:
    def __init__(self, BGP_string:str, ground_truth, predicate_stat: PredicateFeaturesQuery = None, ent_featurizer:EntityFeatures =None):
        self.bgp_string = BGP_string
        triple_strings = BGP_string[1:-1].split(',')
        self.triples = []
        for t in triple_strings:
            self.triples.append(TriplePattern(t, predicate_stat=predicate_stat, ent_featurizer=ent_featurizer))
        if predicate_stat != None:
            self.total_bins = predicate_stat.total_bin
        self.ground_truth = 1 if ground_truth else 0
    
    def set_predicate_feat_gen(self,predicate_stat: PredicateFeaturesQuery):
        self.predicate_stat = predicate_stat
    
    def __str__(self):
        temp_str = 'BGP( '
        for t in self.triples:
            temp_str = temp_str +' '+ str(t)
        temp_str = temp_str +' )'
        return temp_str