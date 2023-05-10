
from graph_construction.bgp import BGP
import networkx as nx

from graph_construction.nodes.node import Node

class tps_graph:
    def __init__(self, bgp:BGP) -> None:
        self.bgp = bgp
        self.graph_creation(bgp)
        self.nodes : list[Node] = []
        self.edges: list[list[int]] = []
        self.graph = nx.DiGraph()
    
    def graph_creation(self, bgp):
        pass