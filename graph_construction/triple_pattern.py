from graph_construction.nodes.node import Node
from feature_extraction.predicates.predicate_features import PredicateFeaturesQuery
from feature_extraction.predicate_features_sub_obj import Predicate_Featurizer_Sub_Obj
from feature_extraction.entity_features import EntityFeatures
from glb_vars import PREDS_W_NO_BIN

class TriplePattern:

    def __init__(self, triple_string:str, node_class = Node):
        splits = triple_string.split(' ')
        splits = [s for s in splits if s != '']
        assert len(splits) == 3
        self.node_class = node_class
        

        self.subject = node_class(splits[0])
        self.subject.nodetype = 0
        self.predicate = node_class(splits[1])
        self.predicate.nodetype = 1
        self.object = node_class(splits[2])
        self.object.nodetype = 2
        
        #self.set_entity_features(ent_featurizer)
        self.subject = self.subject.set_entity_feature()
        self.object = self.object.set_entity_feature()
        self.predicate.set_predicate_features()
        
        """self.predicate.pred_freq = -1
        self.predicate.pred_literals = -1
        self.predicate.pred_subject_count,self.predicate.pred_object_count =0,0
        if predicate_stat != None:
            #self.predicate_stat = predicate_stat
            if self.predicate.type == 'URI':
                self.predicate.bucket = int(predicate_stat.get_bin(self.predicate.node_label))
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
                    self.predicate.pred_subject_count,self.predicate.pred_object_count = predicate_stat.unique_entities_counter[self.predicate.node_label]"""
                
                #for pred_feat,dct in zip([self.predicate.pred_freq,self.predicate.pred_literals,self.predicate.pred_entities],[predicate_stat.predicate_freq,predicate_stat.uniqueLiteralCounter,predicate_stat.unique_entities_counter]):
                #    if self.predicate.node_label in dct.keys():
                #        pred_feat = dct[self.predicate.node_label]
                #    else:
                #        pred_feat = 0
        
    """def set_entity_features(self,ent_featurizer:EntityFeatures):
        if ent_featurizer == None:
            return
        if self.subject.type == 'URI':
            self.subject = self.__set_entity_features_for_node__(self.subject, ent_featurizer)
        if self.object.type == 'URI':
            self.object = self.__set_entity_features_for_node__(self.object, ent_featurizer)
            
            
    def __set_entity_features_for_node__(self,node:Node,ent_featurizer:EntityFeatures):
        bin_no,freq = ent_featurizer.get_feature(node.node_label)
        node.ent_bin = bin_no
        node.ent_freq = freq
        return node"""
        
    
    def __str__(self):
        return f'Triple ({str(self.subject)} {str(self.predicate)} {str(self.object)} )'
    def __eq__(self, other):
        return self.subject == other.subject and self.predicate == other.predicate and self.object == other.object

if __name__ == "__main__":
    t = TriplePattern("?x http://www.wikidata.org/prop/direct/P5395 ?y")
    t2 = TriplePattern("?x http://www.wikidata.org/prop/direct/P5395 ?y")
    print(t.predicate.node_label)