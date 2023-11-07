from graph_construction.node import TriplePattern
from graph_construction.node import Node
class PathNode(TriplePattern):
    def __init__(self, data:dict, node_class=Node):
        self.depthLevel = None
        self.node_class = node_class
        self.predicate = None
        self.path_predicates = list()
        for p in data['Predicates']:
            self.path_predicates.append(node_class(p))
    
