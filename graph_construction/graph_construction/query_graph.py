import json
import os
import dgl
import json5
import networkx as nx
import numpy as np
from graph_construction.stack import Stack
import pandas as pd
from graph_construction.feats.featurizer import FeaturizerBase, FeaturizerPredStats
from graph_construction.node import Node, FilterNode, TriplePattern
from graph_construction.nodes.path_node import PathNode
import torch as th




class QueryPlan:
    max_relations = 16

    def __init__(self, data) -> None:
        self.process(data)
    
    def process(self, data):
        self.level = 0
        self.data = data
        self.triples: list[TriplePattern] = list()
        self.filters: list[FilterNode] = list()
        self.edges = list()

        self.join_vars = {}
        self.filter_dct = {}
        self.op_lvl = {}

        self.iterate_ops(self.add_triple, "Triple")
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

    def initialize_node2_obj(self):
        for t in self.triples:
            if t.id in self.node2obj.keys():
                raise Exception("Existing node createad")
            self.node2obj[t.id] = t
        for f in self.filters:
            if f.id in self.node2obj.keys():
                raise Exception("Existing node createad")
            self.node2obj[f.id] = f

    def iterate_ops(self, func, node_type: str):
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
            if current["opName"] == node_type:
                func(current)

    def iterate_bgp(self, data, func, node_type, filter=None):
        self.level += 1
        for triple in data["subOp"]:
            triple["level"] = self.level
            if triple["opName"] == node_type:
                func(triple, add_data=filter)

    def add_self_loop_triples(self):
        for t in self.triples:
            for v in t.get_joins():
                self.edges.append((t, t, self.get_join_type(t, t, v)))

    def add_triple(self, data, add_data=None):
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

    def add_filters(self, data, add_data=None):
        self.filters
        self.join_vars
        filter_node = FilterNode(data)
        self.filters.append(filter_node)
        filter_triples = QueryPlanUtils.extract_triples(data)
        expr_string = data["expr"]
        expr_string = expr_string.replace(")", " ")
        filt_vars = [x for x in expr_string.split(" ") if x.startswith("?")]
        filter_triples = QueryPlanUtils.map_extracted_triples(
            filter_triples, self.triples
        )
        # dobbel check with filter_node.vars field insead of filt_vars
        for v in filt_vars:
            if v in self.join_vars.keys():
                for t in filter_triples:
                    self.edges.append(
                        (t, filter_node, self.get_join_type(t, filter_node, v))
                    )

    def get_join_type(self, trp1, trp2, common_variable):
        # filter nodes
        if isinstance(trp2, FilterNode):
            return 9

        # s-s
        if trp1.subject == common_variable and trp2.subject == common_variable:
            return 0
        # s-p
        if trp1.subject == common_variable and trp2.predicate == common_variable:
            return 1
        # s-o
        if trp1.subject == common_variable and trp2.object == common_variable:
            return 2

        # p-s
        if trp1.predicate == common_variable and trp2.subject == common_variable:
            return 3
        # p-p
        if trp1.predicate == common_variable and trp2.predicate == common_variable:
            return 4
        # p-o
        if trp1.predicate == common_variable and trp2.object == common_variable:
            return 5
        # o-s
        if trp1.object == common_variable and trp2.subject == common_variable:
            return 6
        # o-p
        if trp1.object == common_variable and trp2.predicate == common_variable:
            return 7
        # o-s
        if trp1.object == common_variable and trp2.object == common_variable:
            return 8

    def assign_trpl_ids(self):
        self.max_id = 0
        for i, t in enumerate(self.triples):
            t.id = i
            self.max_id = np.max([t.id, self.max_id])

    def assign_filt_ids(self):
        for i, t in enumerate(self.filters):
            n_id = 1 + i + self.max_id
            t.id = n_id
            self.max_id = np.max([n_id, self.max_id])

    def add_binaryOP(self, data, add_data=None):
        assert len(data["subOp"]) == 2
        left = data["subOp"][0]
        right = data["subOp"][1]
        left_triples = QueryPlanUtils.extract_triples(left)
        left_triples = QueryPlanUtils.map_extracted_triples(left_triples, self.triples)
        right_triples = QueryPlanUtils.extract_triples(right)
        right_triples = QueryPlanUtils.map_extracted_triples(
            right_triples, self.triples
        )
        for r in right_triples:
            r: TriplePattern
            for l in left_triples:
                l_vars = l.get_joins()
                for r_v in r.get_joins():
                    if r_v in l_vars:
                        # consider adding the other way for union as a special case
                        self.edges.append(
                            (l, r, QueryPlanUtils.get_relations(data["opName"]))
                        )
        # print(left_triples)
        # print("\n\n")
        # print(right_triples)

    def assert_operator(self, data, add_data=None):
        raise Exception("This operator should not exists")

    def networkx(self):
        G = nx.MultiDiGraph()
        for s, t, r in self.edges:
            assert self.node2obj[s.id] == s
            assert self.node2obj[t.id] == t
            G.add_edge(s.id, t.id, rel_type=r)
        return G

    def feature(self, featurizer: FeaturizerBase):
        """_summary_
        Prerequisite for this function is the networkx call (this should happen during initializaition)
        Args:
            featurizer (Featurizer): _description_
        """
        self.node_features = {}
        for n_id in self.nodes:
            self.node_features[n_id] = {
                "node_features": featurizer.featurize(self.node2obj[n_id])
            }

        nx.set_node_attributes(self.G, self.node_features)

    def to_dgl(self):
        """_summary_
        Prerequisite is a call to feature
        Returns:
            _type_: _description_
        """
        try:
            dgl_graph = dgl.from_networkx(
                self.G, edge_attrs=["rel_type"], node_attrs=["node_features"]
            )
        except Exception:
            dgl_graph = dgl.from_networkx(self.G, node_attrs=["node_features"])
            dgl_graph.edata["rel_type"] = th.tensor(np.array([]), dtype=th.int64)
            # print(self.data)
            # print(self.path)
            # exit()
        dgl_graph = dgl.add_self_loop(dgl_graph)
        return dgl_graph

    def get_nodes_in_edges(self):
        nodes = set()
        for node1, node2, _ in self.edges:
            nodes.add(node1.id)
            nodes.add(node2.id)
        return list(nodes)

    def add_sngl_trp_rel(self):
        nodes = self.get_nodes_in_edges()
        for t in self.triples:
            if not t.id in nodes:
                self.edges.append((t, t, 10))

def QueryPlanPath(QueryPlan):
    def __init__(self, data):
        super(QueryPlan,self).__init__(data)
        self.path_nodes: list[PathNode] = list()
    
    def create_bgp(self, bgp):
        """ Method responsible for creating representing the BGP properly
        """

    def create_paths(self, path):
        path_node = PathNode(path)
        self.path_nodes.append(path_node)
        
def test(p, add_data=None):
    pass
    # print(p["level"])


class QueryPlanUtils:
    "filter rel definied in getjointype method"

    def get_relations(op):
        match op:
            case "conditional":
                return 11
            case "leftjoin":
                return 12
            case "join":
                return 13
            case "union":
                return 14
            case "minus":
                return 15
        raise Exception("Operation undefind " + op)

    def extract_triples(data: dict):
        triple_data = []
        stack = Stack()
        stack.push(data)
        while not stack.is_empty():
            current = stack.pop()
            if "subOp" in current.keys():
                for node in reversed(current["subOp"]):
                    stack.push(node)
            if current["opName"] == "Triple":
                triple_data.append(current)
        return triple_data

    def extract_triples_filter(data: dict):
        triple_data = []
        stack = Stack()
        stack.push(data)
        while not stack.is_empty():
            current = stack.pop()
            if "subOp" in current.keys():
                for node in reversed(current["subOp"]):
                    stack.push(node)
            if current["opName"] == "Triple":
                triple_data.append(current)
        return triple_data

    def map_extracted_triples(triple_dct: list[dict], trpl_list: list):
        res_t = list()
        for t in trpl_list:
            if t in triple_dct:
                res_t.append(t)
        return res_t


class QueryPlanCommonBi(QueryPlan):
    """This query plan class only adds relations between binary BGP operations with the triple patterns share the same variables

    Args:
        QueryPlan (_type_): _description_
    """

    max_relations = 16

    def __init__(self, data) -> None:
        super().__init__(data)

    def add_binaryOP(self, data, add_data=None):
        assert len(data["subOp"]) == 2
        left = data["subOp"][0]
        right = data["subOp"][1]
        left_triples = QueryPlanUtils.extract_triples(left)
        left_triples = QueryPlanUtils.map_extracted_triples(left_triples, self.triples)
        right_triples = QueryPlanUtils.extract_triples(right)
        right_triples = QueryPlanUtils.map_extracted_triples(
            right_triples, self.triples
        )
        for r in right_triples:
            for l in left_triples:
                # consider adding the other way for union as a special case
                if self.is_triple_common(l, r):
                    self.edges.append(
                        (r, l, QueryPlanUtils.get_relations(data["opName"]))
                    )

    def is_triple_common(self, l: TriplePattern, r: TriplePattern):
        """checks if the triple patterns are joinable

        Args:
            l (TriplePattern): left triple pattern
            r (TriplePattern): right triple pattern
        """
        for lvar in l.get_joins():
            for rvar in r.get_joins():
                if lvar == rvar:
                    return True
        return False

    def add_self_loop_triples(self):
        for t in self.triples:
            self.edges.append((t, t, 15))


def create_query_plan(path, query_plan=QueryPlan):
    try:
        data = json.load(open(path, "r"))
    except Exception:
        data = json5.load(open(path, "r"))
    q = query_plan(data)
    q.path = path
    return q


def create_query_plans_dir(
    source_dir, ids: set[str] = None, add_id=False, query_plan=QueryPlanCommonBi
):
    if ids is None:
        # TODO: check whether lsqQuery is necessary
        files = [x for x in os.listdir(source_dir) if x.startswith("lsqQuery")]
    else:
        ids = [str(x) for x in ids]
        files = [x for x in os.listdir(source_dir) if x in ids]
    if add_id:
        return [
            (create_query_plan(f"{source_dir}{x}", query_plan=query_plan), x)
            for x in files
        ]

    query_plans = [
        create_query_plan(f"{source_dir}{x}", query_plan=query_plan) for x in files
    ]
    return query_plans


def create_dgl_graphs(
    qps: list[QueryPlan], featurizer: FeaturizerBase, without_id=True
) -> list:
    if without_id:
        return create_dgl_graph_helper(qps, featurizer)

    dgl_graphs = list()
    for x, id in qps:
        x: QueryPlan
        if len(x.nodes) == 0:
            continue
        x.feature(featurizer)
        dgl_graph = x.to_dgl()
        dgl_graphs.append((dgl_graph, id))
    return dgl_graphs


def create_dgl_graph_helper(qps: list[QueryPlan], featurizer: FeaturizerBase) -> list:
    dgl_graphs = list()
    for x in qps:
        # for now skip single triple patterns
        if len(x.edges) <= 0:
            continue
        x.feature(featurizer)
        dgl_graph = x.to_dgl()
        dgl_graphs.append(dgl_graph)
    return dgl_graphs


def create_query_graphs_data_split(
    source_dir,
    query_path="/qpp/dataset/DBpedia_2016_12k_sample/train_sampled.tsv",
    query_plan=QueryPlan,
    is_lsq=False,
    feat: FeaturizerBase = None,
):
    df = pd.read_csv(query_path, sep="\t")
    if is_lsq:
        ids = set([x[20:] for x in df["queryID"]])
    else:
        ids = set([x for x in df["queryID"]])

    qps = create_query_plans_dir(source_dir, ids, query_plan=query_plan, add_id=True)
    for qp, id in qps:
        try:
            assert len(qp.G.nodes) > 0
        except AssertionError:
            print(qp.data)
            print(qp.path)
            exit()
    return create_dgl_graphs(qps, feat, without_id=False)


def query_graphs_with_lats(
    source_dir,
    query_path="/qpp/dataset/DBpedia_2016_12k_sample/train_sampled.tsv",
    feat: FeaturizerBase = None,
    query_plan=QueryPlan,
    time_col="mean_latency",
    is_lsq=False,
):
    df = pd.read_csv(query_path, sep="\t")
    if is_lsq:
        df["queryID"] = df["queryID"].apply(lambda x: x[20:])
    df.set_index("queryID", inplace=True)
    # print(df.loc["lsqQuery-UBZNr7M1ITVUf21mrBIQ9W4f6cdpJr6DQbr0HkWKOnw"][time_col])

    graphs_ids = create_query_graphs_data_split(
        source_dir,
        query_path=query_path,
        feat=feat,
        query_plan=query_plan,
        is_lsq=is_lsq,
    )
    samples = list()
    for g, id in graphs_ids:
        if not is_lsq:
            id = int(id)
        lat = df.loc[id][time_col]
        if isinstance(lat, pd.Series):
            lat = lat.iloc[0]
        samples.append((g, id, lat))
    return samples


def query_graph_w_class_vec(
    source_dir,
    query_path="/qpp/dataset/DBpedia_2016_12k_sample/train_sampled.tsv",
    feat: FeaturizerBase = None,
    time_col="mean_latency",
    cls_funct=lambda x: x,
    query_plan=QueryPlanCommonBi,
    is_lsq=False,
):
    samples = query_graphs_with_lats(
        source_dir=source_dir,
        query_path=query_path,
        feat=feat,
        time_col=time_col,
        query_plan=query_plan,
        is_lsq=is_lsq,
    )
    return query_graph_w_class_vec_helper(samples, cls_funct)


def snap_lat2onehot(lat):
    vec = np.zeros(6)
    if lat < 0.01:
        vec[0] = 1
    elif (0.01 < lat) and (lat < 0.1):
        vec[1] = 1
    elif (0.1 < lat) and (lat < 1):
        vec[2] = 1
    elif (1 < lat) and (lat < 10):
        vec[3] = 1
    elif 10 < lat and lat < 100:
        vec[4] = 1
    elif lat > 100:
        vec[5] = 1

    return vec


def snap_lat2onehotv2(lat):
    vec = np.zeros(3)
    if lat < 1:
        vec[0] = 1
    elif (1 < lat) and (lat < 10):
        vec[1] = 1
    elif 10 < lat:
        vec[2] = 1
    return vec


def snap_lat_2onehot_4_cat(lat):
    vec = np.zeros(4)
    if lat < 0.3:
        vec[0] = 1
    elif (0.3 < lat) and (lat < 1):
        vec[1] = 1
    elif (1 < lat) and (lat < 10):
        vec[1] = 1
    elif 10 < lat:
        vec[2] = 1
    return vec


def snap_lat2onehot_binary(lat):
    vec = np.zeros(2)
    if lat < 1:
        vec[0] = 1
    else:
        vec[1] = 1
    return vec


def query_graph_w_class_vec_helper(samples: list[tuple], cls_funct):
    graphs = []
    clas_list = []
    ids = []
    for g, id, lat in samples:
        graphs.append(g)
        ids.append(id)
        try:
            clas_list.append(cls_funct(lat))
        except Exception:
            print(lat)
            print("Something went wrong")
            exit()
    clas_list = th.tensor(np.array(clas_list), dtype=th.float32)
    return graphs, clas_list, ids


if __name__ == "__main__":
    # feat = FeaturizerBase(5)
    pred_stat_path = (
        "/PlanRGCN/extracted_features_dbpedia2016/predicate/pred_stat/batches_response_stats"
    )
    feat = FeaturizerPredStats(pred_stat_path)
    q = create_query_plan(
        "/query_plans_dbpedia/lsqQuery---GN1alrTxD0-fWJkepMSVXeW1wiZ68OlE_ASH5d5XM"
    )
    #print(q.edges)
    q.feature(feat)
    G = q.G
    dgl_g: dgl.DGLGraph = q.to_dgl()
    print("dgl" + str(dgl_g.all_edges()))

    exit()
    """query_plan = create_query_plan(
        "/PlanRGCN/extracted_features/queryplans/lsqQuery-Yi0-ewp6Py0hTrLTppeXWTUNGW3z2KhaNz1wHKDCdMw"
    )
    query_plan.feature(feat)
    dgl_grpah = query_plan.to_dgl()"""
    """query_graphs_with_lats(
        "/PlanRGCN/extracted_features/queryplans/",
        query_path="/qpp/dataset/DBpedia_2016_12k_sample/train_sampled.tsv",
        feat=feat,
        time_col="mean_latency",
    )"""
    # qps = create_query_plans_dir("/PlanRGCN/extracted_features/queryplans/")
    # print(len(create_dgl_graphs(qps, feat)))
# q.extract_triples()
