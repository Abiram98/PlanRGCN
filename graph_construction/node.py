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
    
    def __str__(self):
        if self.type == None:
            return self.node_label
        else:
            return f'{self.type} {self.node_label}'
    def __eq__(self, other):
        return self.node_label == other.node_label
    def __hash__(self) -> int:
        return hash(self.node_label)
    
    def get_features(self, pred_bins=30):
        nodetype = np.zeros(4)
        nodetype[self.nodetype] = 1
        predicate_features = np.zeros(3)
        predicate_bins = np.zeros(pred_bins)
        if self.nodetype == 1:
            predicate_features[0] = self.pred_freq
            predicate_features[1] = self.pred_literals
            predicate_features[2] = self.pred_entities
            predicate_bins[self.bucket] = 1
        return np.concatenate((nodetype ,predicate_features, predicate_bins))