import json
import os
import dgl
import networkx as nx
import numpy as np
from graph_construction.stack import Stack
import pandas as pd
from graph_construction.featurizer import FeaturizerBase


class Node:
    pred_bins = 30
    pred_topk = 15
    pred_feat_sub_obj_no: bool = None
    use_ent_feat: bool = False
    ent_bins: int = None
    use_join_features: bool = True

    def __init__(self, node_label: str) -> None:
        self.node_label = node_label
        if node_label.startswith("?"):
            self.type = "VAR"
        elif node_label.startswith("<http") or node_label.startswith("http"):
            self.type = "URI"
        elif node_label.startswith("join"):
            self.type = "JOIN"
        else:
            self.type = None

        self.pred_freq = None
        self.pred_literals = None
        self.pred_entities = None
        # self.topK = None

        # for join node
        self.is_subject_var = None
        self.is_pred_var = None
        self.is_object_var = None

    def __str__(self):
        if self.type == None:
            return self.node_label
        else:
            return f"{self.type} {self.node_label}"

    def __eq__(self, other):
        if isinstance(other, str):
            return self.node_label == other
        return self.node_label == other.node_label

    def __hash__(self) -> int:
        return hash(self.node_label)

    def get_pred_features(self):
        pred_bins, pred_topk, pred_feat_sub_obj_no = (
            Node.pred_bins,
            Node.pred_topk,
            Node.pred_feaurizer,
        )

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
                predicate_bins[pred_bins - 1] = 1
            if self.topK != None and self.topK < pred_topk:
                topk_vec[self.topK] = 1
        if np.sum(np.isnan(predicate_features)) > 0:
            predicate_features[np.isnan(predicate_features)] = 0
            # raise Exception
        if np.sum(np.isnan(predicate_bins)) > 0:
            raise Exception
        if np.sum(np.isnan(topk_vec)) > 0:
            raise Exception
        return np.concatenate((predicate_features, predicate_bins, topk_vec))
        # return predicate_features,predicate_bins, topk_vec

    def get_ent_features(self, ent_bins):
        freq_vec_ent = np.zeros(1)
        ent_bins_vec = np.zeros(ent_bins + 1)
        if self.nodetype in [0, 2] and self.type == "URI":
            freq_vec_ent[0] = self.ent_freq
            ent_bins_vec[self.ent_bin] = 1
        if np.sum(np.isnan(freq_vec_ent)) > 0:
            raise Exception
        if np.sum(np.isnan(ent_bins_vec)) > 0:
            raise Exception
        return np.concatenate((freq_vec_ent, ent_bins_vec))

    def get_join_features(self):
        join_feat = np.zeros(3)
        if self.nodetype == 3:
            return join_feat
        if self.is_subject_var:
            join_feat[0] = 1
        if self.is_pred_var:
            join_feat[1] = 1
        if self.is_object_var:
            join_feat[2] = 1
        return join_feat

    def get_features(self):
        nodetype = np.zeros(4)
        nodetype[self.nodetype] = 1
        predicate_features = self.get_pred_features()
        if Node.use_ent_feat:
            ent_features = self.get_ent_features(Node.ent_bins)
        else:
            ent_features = np.array([])
        if Node.use_join_features:
            join_feat = self.get_join_features()
        else:
            join_feat = np.array([])
        return np.concatenate((nodetype, join_feat, predicate_features, ent_features))

    def set_predicate_features(self):
        self.pred_freq = -1
        self.pred_literals = -1
        self.pred_subject_count, self.pred_object_count = -1, -1


class FilterNode:
    def __init__(self, data) -> None:
        self.data = data
        self.expr_string = data["expr"]
        splts = data["expr"].split(" ")
        self.vars = [x for x in splts if x.startswith("?")]
        for i in range(len(self.vars)):
            if ")" in self.vars[i]:
                self.vars[i] = self.vars[i].split(")")[0]

    def __str__(self) -> str:
        return f"Filter node : {self.expr_string}"

    def __hash__(self) -> int:
        return self.data["expr"].__hash__()


class QueryPlan:
    def __init__(self, data) -> None:
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
        self.add_self_loop_triples()
        self.assign_trpl_ids()
        self.assign_filt_ids()

        # filt_edge = [(d.id, b.id, r) for (d, b, r) in self.edges if r == 9]
        # print(filt_edge)
        self.iterate_ops(self.add_binaryOP, "minus")
        self.iterate_ops(self.add_binaryOP, "union")
        self.iterate_ops(self.add_binaryOP, "leftjoin")
        self.iterate_ops(self.add_binaryOP, "conditional")
        self.iterate_ops(self.add_binaryOP, "join")
        # can be used to check for existence of operators
        # self.iterate_ops(self.assert_operator, "leftjoin")
        # self.iterate_ops(self.assert_operator, "join")
        self.iterate_ops(self.assert_operator, "diff")
        self.iterate_ops(self.assert_operator, "lateral")

        # print(self.edges)
        self.nodes = [x.id for x in self.triples]
        self.nodes.extend([x.id for x in self.filters])
        self.node2obj = {}
        self.initialize_node2_obj()
        self.G = self.networkx()

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
        filterdata = None
        while not stack.is_empty():
            current = stack.pop()
            if "subOp" in current:
                if current["opName"] == "filter":
                    filterdata = FilterNode(current)
                    self.filter_dct[filterdata] = []
                if current["opName"] == "BGP":
                    self.iterate_bgp(current, func, node_type, filter=filterdata)
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
        filter_triples = QueryPlanUtils.map_extracted_triples(
            filter_triples, self.triples
        )
        for v in filter_node.vars:
            if v in self.join_vars.keys():
                for t in self.join_vars[v]:
                    if t in filter_triples:
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
            for l in left_triples:
                # consider adding the other way for union as a special case
                self.edges.append((r, l, QueryPlanUtils.get_relations(data["opName"])))
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
        dgl_graph = dgl.from_networkx(
            self.G, edge_attrs=["rel_type"], node_attrs=["node_features"]
        )
        # dgl_graph = dgl.add_self_loop(dgl_graph)
        return dgl_graph


def test(p, add_data=None):
    pass
    # print(p["level"])


class QueryPlanUtils:
    def get_relations(op):
        match op:
            case "conditional":
                return 10
            case "union":
                return 11
            case "minus":
                return 12
            case "leftjoin":
                return 13
            case "join":
                return 14
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

    def map_extracted_triples(triple_dct: list[dict], trpl_list: list):
        res_t = list()
        for t in trpl_list:
            if t in triple_dct:
                res_t.append(t)
        return res_t


def create_query_plan(path):
    data = json.load(open(path, "r"))
    q = QueryPlan(data)
    q.path = path
    return q


class TriplePattern:
    """Class representing a triple pattern. Joins on constants are not considered"""

    def __init__(self, data: dict, node_class=Node):
        self.depthLevel = None
        self.node_class = node_class

        self.subject = node_class(data["Subject"])
        self.subject.nodetype = 0
        self.predicate = node_class(data["Predicate"])
        self.predicate.nodetype = 1
        self.object = node_class(data["Object"])
        self.object.nodetype = 2
        if "level" in data.keys():
            self.level = data["level"]

    def __hash__(self):
        return hash(self.subject) + hash(self.predicate) + hash(self.object)

    def __str__(self):
        return f"Triple ({str(self.subject)} {str(self.predicate)} {str(self.object)} )"

    def __repr__(self):
        return f"Triple ({str(self.subject)} {str(self.predicate)} {str(self.object)} )"

    def __eq__(self, other):
        if isinstance(other, dict):
            return (
                self.subject.node_label == other["Subject"]
                and self.predicate.node_label == other["Predicate"]
                and self.object.node_label == other["Object"]
            )
        return (
            self.subject == other.subject
            and self.predicate == other.predicate
            and self.object == other.object
        )

    def get_variables(self):
        v = []
        v.append(self.subject)
        v.append(self.predicate)
        v.append(self.object)
        """if self.subject.type == "VAR":
            v.append(self.subject)
        if self.predicate.type == "VAR":
            v.append(self.predicate)
        if self.object.type == "VAR":
            v.append(self.object)"""
        return v

    def get_joins(self):
        return list(set(self.get_variables()))


def create_query_plans_dir(source_dir, ids: set[str] = None, add_id=False):
    if ids is None:
        files = [x for x in os.listdir(source_dir) if x.startswith("lsqQuery")]
    else:
        files = [
            x for x in os.listdir(source_dir) if x.startswith("lsqQuery") and x in ids
        ]
    if add_id:
        return [(create_query_plan(f"{source_dir}{x}"), x) for x in files]

    query_plans = [create_query_plan(f"{source_dir}{x}") for x in files]
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
    feat: FeaturizerBase = None,
):
    df = pd.read_csv(query_path, sep="\t")
    ids = set([x[20:] for x in df["queryID"]])
    qps = create_query_plans_dir(source_dir, ids, add_id=True)
    return create_dgl_graphs(qps, feat, without_id=False)


def query_graphs_with_lats(
    source_dir,
    query_path="/qpp/dataset/DBpedia_2016_12k_sample/train_sampled.tsv",
    feat: FeaturizerBase = None,
    time_col="mean_latency",
):
    df = pd.read_csv(query_path, sep="\t")
    df["queryID"] = df["queryID"].apply(lambda x: x[20:])
    df.set_index("queryID", inplace=True)
    # print(df.loc["lsqQuery-UBZNr7M1ITVUf21mrBIQ9W4f6cdpJr6DQbr0HkWKOnw"][time_col])

    graphs_ids = create_query_graphs_data_split(
        source_dir, query_path=query_path, feat=feat
    )
    samples = list()
    for g, id in graphs_ids:
        lat = df.loc[id][time_col]
        samples.append((g, id, lat))
    return samples


if __name__ == "__main__":
    feat = FeaturizerBase(5)
    """q = create_query_plan(
        "/PlanRGCN/extracted_features/queryplans/lsqQuery-TaZVkkyiv_-35SZOipT1c9ppAwnh_6t_JeNMDYGRFGk"
    )
    print(q.edges)
    q.feature(feat)
    G = q.G
    dgl_g: dgl.DGLGraph = q.to_dgl()
    print("dgl" + str(dgl_g.all_edges()))
    print("dgl" + str(dgl_g.all_edges()))

    exit()"""
    """query_plan = create_query_plan(
        "/PlanRGCN/extracted_features/queryplans/lsqQuery-Yi0-ewp6Py0hTrLTppeXWTUNGW3z2KhaNz1wHKDCdMw"
    )
    query_plan.feature(feat)
    dgl_grpah = query_plan.to_dgl()"""
    query_graphs_with_lats(
        "/PlanRGCN/extracted_features/queryplans/",
        query_path="/qpp/dataset/DBpedia_2016_12k_sample/train_sampled.tsv",
        feat=feat,
        time_col="mean_latency",
    )
    # qps = create_query_plans_dir("/PlanRGCN/extracted_features/queryplans/")
    # print(len(create_dgl_graphs(qps, feat)))
# q.extract_triples()
