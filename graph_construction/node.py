import numpy as np
import numpy as np

class Node:
    def __init__(self, node_label:str) -> None:
        self.node_label = node_label
        if node_label.startswith('?'):
            self.type = 'VAR'
        elif node_label.startswith('http'):
            self.type = 'URI'
        elif node_label.startswith('join'):
            self.type = 'JOIN'
        else:
            self.type = None
        
        self.pred_freq = None
        self.pred_literals = None
        self.pred_entities= None
        #self.topK = None
    
    def __str__(self):
        if self.type == None:
            return self.node_label
        else:
            return f'{self.type} {self.node_label}'
    def __eq__(self, other):
        return self.node_label == other.node_label
    def __hash__(self) -> int:
        return hash(self.node_label)
    
    def get_pred_features(self, pred_bins, pred_topk,pred_feat_sub_obj_no):
        predicate_bins = np.zeros(pred_bins)
        topk_vec = np.zeros(pred_topk)
        
        if pred_feat_sub_obj_no:
            predicate_features = np.zeros(4)
        else:
            predicate_features = np.zeros(3)
        if self.nodetype == 1:
            predicate_features[0] = self.pred_freq
            predicate_features[1] = self.pred_literals
            if not pred_feat_sub_obj_no:
                predicate_features[2] = self.pred_entities
            else:
                predicate_features[2] = self.pred_subject_count
                predicate_features[3] = self.pred_object_count
            try:
                predicate_bins[self.bucket] = 1
            except AttributeError:
                predicate_bins[pred_bins-1] = 1
            if self.topK != None and self.topK < pred_topk:
                topk_vec[self.topK] = 1
        if np.sum(np.isnan( predicate_features)) > 0:
            predicate_features[np.isnan(predicate_features)] = 0
            #raise Exception
        if np.sum(np.isnan( predicate_bins)) > 0:
            raise Exception
        if np.sum(np.isnan( topk_vec)) > 0:
            raise Exception
        return predicate_features,predicate_bins, topk_vec
    def get_ent_features(self, ent_bins):
        freq_vec_ent = np.zeros(1)
        ent_bins_vec = np.zeros(ent_bins+1)
        if self.nodetype in [0,2] and self.type == 'URI':
            freq_vec_ent[0] = self.ent_freq
            ent_bins_vec[self.ent_bin] = 1
        if np.sum(np.isnan( freq_vec_ent)) > 0:
            raise Exception
        if np.sum(np.isnan( ent_bins_vec)) > 0:
            raise Exception
        return np.concatenate((freq_vec_ent,ent_bins_vec))
    
    def get_features(self, pred_bins=30, pred_topk=15,pred_feat_sub_obj_no=True, use_ent_feat=False, ent_bins = None):
        nodetype = np.zeros(4)
        nodetype[self.nodetype] = 1
        predicate_features,predicate_bins, topk_vec = self.get_pred_features(pred_bins, pred_topk,pred_feat_sub_obj_no)
        if use_ent_feat:
            ent_features = self.get_ent_features(ent_bins)
        else:
            ent_features = np.array([])
        return np.concatenate((nodetype ,predicate_features, predicate_bins, topk_vec, ent_features))