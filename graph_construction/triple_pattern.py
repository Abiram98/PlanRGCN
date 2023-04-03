from graph_construction.node import Node
from feature_extraction.predicate_features import PredicateFeaturesQuery
from feature_extraction.predicate_features_sub_obj import Predicate_Featurizer_Sub_Obj
from glb_vars import PREDS_W_NO_BIN

class TriplePattern:
    def __init__(self, triple_string:str, predicate_stat:PredicateFeaturesQuery = None):
        splits = triple_string.split(' ')
        splits = [s for s in splits if s != '']
        assert len(splits) == 3
     
        self.subject = Node(splits[0])
        self.subject.nodetype = 0
        self.predicate = Node(splits[1])
        self.predicate.nodetype = 1
        self.object = Node(splits[2])
        self.object.nodetype = 2
        
        self.predicate.pred_freq = -1
        self.predicate.pred_freq = -1
        self.predicate.pred_literals = -1
        
        if predicate_stat != None:
            #self.predicate_stat = predicate_stat
            if self.predicate.type == 'URI':
                self.predicate.bucket = predicate_stat.get_bin(self.predicate.node_label)
                self.predicate.topK = predicate_stat.top_k_predicate(self.predicate.node_label)
                if self.predicate.bucket == None:
                    PREDS_W_NO_BIN.append(self.predicate)
                    self.predicate.bucket = 0
                
                if self.predicate.node_label in predicate_stat.predicate_freq.keys():
                    self.predicate.pred_freq = predicate_stat.predicate_freq[self.predicate.node_label]
                    
                
                if self.predicate.node_label in predicate_stat.uniqueLiteralCounter.keys():
                    self.predicate.pred_literals = predicate_stat.uniqueLiteralCounter[self.predicate.node_label]
                    
                
                if (not isinstance(predicate_stat,Predicate_Featurizer_Sub_Obj)) and (self.predicate.node_label in predicate_stat.unique_entities_counter.keys()):
                    self.predicate.pred_entities = predicate_stat.unique_entities_counter[self.predicate.node_label]
                
                if (isinstance(predicate_stat,Predicate_Featurizer_Sub_Obj)) and (self.predicate.node_label in predicate_stat.unique_entities_counter.keys()):
                    self.predicate.pred_subject_count,self.predicate.pred_object_count = predicate_stat.unique_entities_counter[self.predicate.node_label]
                
                #for pred_feat,dct in zip([self.predicate.pred_freq,self.predicate.pred_literals,self.predicate.pred_entities],[predicate_stat.predicate_freq,predicate_stat.uniqueLiteralCounter,predicate_stat.unique_entities_counter]):
                #    if self.predicate.node_label in dct.keys():
                #        pred_feat = dct[self.predicate.node_label]
                #    else:
                #        pred_feat = 0
    
    def __str__(self):
        return f'Triple ({str(self.subject)} {str(self.predicate)} {str(self.object)} )'
    def __eq__(self, other):
        return self.subject == other.subject and self.predicate == other.predicate and self.object == other.object