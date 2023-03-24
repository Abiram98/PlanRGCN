import json
import networkx as nx, numpy as np
from feature_extraction.predicate_features import PredicateFeaturesQuery
class BGP:
    def __init__(self, BGP_string:str, ground_truth, predicate_stat: PredicateFeaturesQuery = None):
        triple_strings = BGP_string[1:-1].split(',')
        self.triples = []
        for t in triple_strings:
            self.triples.append(TriplePattern(t, predicate_stat))
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
        
        if predicate_stat != None:
            self.predicate_stat = predicate_stat
            if self.predicate.type == 'URI':
                self.predicate.bucket = predicate_stat.get_bin(self.predicate.node_label)
    
    def __str__(self):
        return f'Triple ({str(self.subject)} {str(self.predicate)} {str(self.object)} )'
    def __eq__(self, other):
        return self.subject == other.subject and self.predicate == other.predicate and self.object == other.object

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
        
        self.pred_freq = 0
        self.pred_literals = 0
        self.pred_entities= 0
    def __str__(self):
        if self.type == None:
            return self.node_label
        else:
            return f'{self.type} {self.node_label}'
    def __eq__(self, other):
        return self.node_label == other.node_label
    def __hash__(self) -> int:
        return hash(self.node_label)
    
    def get_features(self, pred_bins):
        nodetype = np.zeros(4)
        nodetype[self.nodetype] = 1
        predicate_features = np.zeros(3)
        predicate_bins = np.zeros(pred_bins)
        if self.nodetype == 1:
            predicate_features[0] = self.pred_freq
            predicate_features[1] = self.pred_literals
            predicate_features[2] = self.pred_entities
            pred_bins[self.bucket] = 1
        return np.concatenate((nodetype ,predicate_features, pred_bins))

def load_BGPS_from_json(path):
    data = None
    with open(path,'rb') as f:
        data = json.load(f)
                
    if data == None:
        print('Data could not be loaded!')
        return
    BGP_strings = list(data.keys())
    BGPs = []
    for bgp_string in BGP_strings:
        BGPs.append(BGP(bgp_string, data[bgp_string]))
    return BGPs

def get_predicates(bgps: list):
    predicates = set()
    for bgp in bgps:
        for triple in bgp.triples:
            if triple.predicate.type == 'URI':
                predicates.add(triple.predicate)
    return list(predicates)

def get_entities(bgps: list):
    entities = set()
    for bgp in bgps:
        for triple in bgp.triples:
            for e in [triple.subject, triple.object]:
                if e.type == 'URI':
                    entities.add(e)
    return list(entities)

class BGPGraph:
    def __init__(self, bgp : BGP):
        self.bgp = bgp
        self.nodes = []
        self.edges = []
        self.graph = nx.DiGraph()
        self.node_to_idx = {}
        self.current_id = 0
        self.join_id = 0
        
    def create_graph(self):
        prev_join = None
        for trp in self.bgp.triples:
            subject_id = self.node_id(trp.subject)
            predicate_id = self.node_id(trp.predicate)
            object_id = self.node_id(trp.object)
            self.graph.add_edge(subject_id,predicate_id)
            self.graph.add_edge(predicate_id, object_id)
            
            join = self.create_join_node()
            join_id = self.node_id(join)
            self.graph.add_edge(subject_id,join_id)
            self.graph.add_edge(predicate_id,join_id)
            self.graph.add_edge(object_id,join_id)
            if prev_join != None:
                self.graph.add_edge(prev_join,join_id)
                prev_join = join_id
            
    
    def node_id(self, node: Node):
        if node in self.node_to_idx.keys():
            return self.node_to_idx[node]
        self.node_to_idx[node] = self.current_id
        self.current_id += 1
        self.nodes.append(node)
        return self.current_id-1
    
    def create_join_node(self):
        join_node = Node(f"join{self.join_id}")
        join_node.nodetype = 3
        self.join_id += 1
        return join_node
    
    def get_edge_list(self):
        in_vertex, out_vertex = [],[]
        for (x,y) in self.graph.edges():
            in_vertex.append(x)
            out_vertex.append(y)
        return in_vertex,out_vertex
    def get_node_representation(self):
        return np.stack([x.get_features(pred_bins) for x in self.nodes])
    

if __name__ == "__main__":
    bgps = load_BGPS_from_json('/work/data/train_data.json')
    print(f'BGPS loaded : {len(bgps)}')
    bgp_g = BGPGraph(bgps[0])
    bgp_g.create_graph()
    print(bgp_g.get_node_representation())
    #ents = get_entities(bgps)
    #preds = get_predicates(bgps)

    #print(f'Entities extracted: {len(ents)}')
    #print(f'Preds extracted: {len(preds)}')