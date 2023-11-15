from graph_construction.node import FilterNode, TriplePattern
from graph_construction.nodes.path_node import PathNode
from graph_construction.qp.query_plan import QueryPlan
from torch import Stack


class QueryPlanPath(QueryPlan):
    def __init__(self, data):
        super(QueryPlan, self).__init__(data)

    def process(self, data):
        self.level = 0
        self.data = data
        self.triples: list[TriplePattern] = list()
        self.filters: list[FilterNode] = list()
        self.path_nodes: list[PathNode] = list()
        self.edges = list()

        self.join_vars = {}
        self.filter_dct = {}
        self.op_lvl = {}

        self.iterate_ops(self.add_triple, "Triple")
        self.iterate_ops(self.add_path, "path")
        self.iterate_ops(self.add_filters, "filter")
        # self.add_self_loop_triples()
        self.assign_trpl_ids()
        self.assign_filt_ids()

        # filt_edge = [(d.id, b.id, r) for (d, b, r) in self.edges if r == 9]
        # print(filt_edge)
        # self.iterate_ops(self.add_binaryOP, "minus")
        # self.iterate_ops(self.add_binaryOP, "union")
        self.iterate_ops(self.add_binaryOP, "join")
        self.iterate_ops(self.add_binaryOP, "leftjoin")
        self.iterate_ops(self.add_binaryOP, "conditional")
        self.add_sngl_trp_rel()
        # can be used to check for existence of operators
        # self.iterate_ops(self.assert_operator, "leftjoin")
        # self.iterate_ops(self.assert_operator, "join")
        # self.iterate_ops(self.assert_operator, "diff")
        # self.iterate_ops(self.assert_operator, "lateral")

        # print(self.edges)
        self.nodes = [x.id for x in self.triples]
        self.nodes.extend([x.id for x in self.filters])

        self.node2obj = {}
        self.initialize_node2_obj()
        self.G = self.networkx()
        self.G.add_nodes_from(self.nodes)
    
    
    def iterate(self, func):
        current = self.data
        current["level"] = self.level
        if current == None:
            return
        stack = Stack()
        stack.push(current)
        while not stack.is_empty():
            current = stack.pop()
            if "subOp" in current:
                if current["opName"] == "BGP":
                    self.iterate_bgp(current, func, node_type, filter=None)
                else:
                    self.level += 1
                    for node in reversed(current["subOp"]):
                        node["level"] = self.level
                        stack.push(node)
            func(current)
    
    def add_tripleOrPath(self, data, add_data=None):
        if data['opName'] == "path":
            t = PathNode(data)
        else:
            t = TriplePattern(data)
        join_v = t.get_joins()
        for v in join_v:
            if v in self.join_vars.keys():
                triple_lst = self.join_vars[v]
                for tp in triple_lst:
                    self.edges.append((tp, t, self.get_join_type(tp, t, v)))
                self.join_vars[v].append(t)
            else:
                self.join_vars[v] = [t]
        self.triples.append(t)
        """if add_data != None:
            add_data: FilterNode
            for v in add_data.vars:
                t_var_labels = [tv.node_label for tv in t.get_joins()]
                if v in t_var_labels:
                    print(t)"""
    def create_paths(self, path):
        path_node = PathNode(path)
        self.path_nodes.append(path_node)