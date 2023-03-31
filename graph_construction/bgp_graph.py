from graph_construction.bgp import BGP
import networkx as nx
from graph_construction.node import Node
import numpy as np


class BGPGraph:
    def __init__(self, bgp : BGP):
        self.bgp : BGP = bgp
        self.nodes : list[Node] = []
        self.edges: list[list[int]] = []
        self.graph = nx.DiGraph()
        self.node_to_idx = {}
        self.current_id = 0
        self.join_id = 0
        self.last_join_index = None
        self.vars = {}
        self.gt = self.bgp.ground_truth
        
        self.create_graph()
        
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
            self.last_join_index = join_id
            self.graph.add_edge(subject_id,join_id)
            self.graph.add_edge(predicate_id,join_id)
            self.graph.add_edge(object_id,join_id)
            if prev_join != None:
                self.graph.add_edge(prev_join,join_id)
                prev_join = join_id
        self.add_var_edges()
            
    
    
    def node_id(self, node: Node, variable_creation=True):
        if variable_creation:
            node = self.variable_node_check(node)
        if node in self.node_to_idx.keys():
            return self.node_to_idx[node]
        self.node_to_idx[node] = self.current_id
        self.current_id += 1
        self.nodes.append(node)
        return self.current_id-1
    
    def variable_node_check(self, node:Node):
        if node.type == 'VAR':
            node2 = node
            if not node.node_label in self.vars.keys():
                node2.node_label = node2.node_label + '0'
                self.vars[node.node_label] = [node2]
                return node2
            else:
                node2.node_label = node2.node_label + str(len(self.vars[node]))
                self.vars[node.node_label].append(node2)
                return node2
        return node
    
    def add_var_edges(self):
        for var in self.vars.keys():
            for i in range(len(self.vars[var])):
                if i == 0:
                    continue
                current_id = self.node_id(self.vars[var][i],variable_creation=False)
                for j in range(i):
                    prev = self.node_id(self.vars[var][j],variable_creation=False)
                    self.graph.add_edge(prev,current_id)
        
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
    
    def get_node_representation(self, pred_bins):
        return np.stack([x.get_features(pred_bins) for x in self.nodes])