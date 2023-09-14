import json
import os
import pickle
import numpy as np
from graph_construction.node import Node, FilterNode, TriplePattern


class FeaturizerBase:
    def __init__(self, vec_size) -> None:
        self.vec_size = vec_size

    def featurize(self, node):
        return np.array([1, 0, 0, 0, 0])


class FeaturizerPredStats(FeaturizerBase):
    def __init__(
        self,
        pred_stat_path="/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats",
    ) -> None:
        self.pred_stat_path = pred_stat_path

        p = PredStats(path=pred_stat_path)
        self.pred_freq = p.triple_freq
        self.pred_ents = p.pred_ents
        self.pred_lits = p.pred_lits

        self.filter_size = 6
        self.tp_size = 6

    def featurize(self, node):
        if isinstance(node, FilterNode):
            return self.filter_features(node).astype("float64")
        elif isinstance(node, TriplePattern):
            return self.tp_features(node).astype("float64")
        else:
            raise Exception("unknown node type")

    def filter_features(self, node):
        # vec = np.zeros(6)
        vec = list()
        for f in [
            FilterFeatureUtils.isLogical,
            FilterFeatureUtils.isArithmetic,
            FilterFeatureUtils.isComparison,
            FilterFeatureUtils.isGeneralFunction,
            FilterFeatureUtils.isStringManipulation,
            FilterFeatureUtils.isTime,
        ]:
            if f(node.expr_string):
                vec.append(1)
            else:
                vec.append(0)
        return np.concatenate((np.zeros(self.tp_size), np.array(vec)), axis=0)

    def tp_features(self, node):
        var_vec = np.array(
            [node.subject.nodetype, node.predicate.nodetype, node.object.nodetype]
        )
        freq = self.get_value_dict(self.pred_freq, node.predicate.node_label)
        lits = self.get_value_dict(self.pred_lits, node.predicate.node_label)
        ents = self.get_value_dict(self.pred_ents, node.predicate.node_label)

        stat_vec = np.array([freq, lits, ents])
        return np.concatenate((var_vec, stat_vec, np.zeros(self.filter_size)), axis=0)

    def get_value_dict(self, dct: dict, predicate):
        try:
            return dct[predicate]
        except KeyError:
            return 0


class FeaturizerPredCo(FeaturizerPredStats):
    def __init__(
        self,
        pred_stat_path="/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats",
        pred_com_path="/PlanRGCN/data/pred/pred_co/pred2index_louvain.pickle",
    ) -> None:
        super().__init__(pred_stat_path)

        self.pred2index, self.max_pred = pickle.load(open(pred_com_path, "rb"))
        self.tp_size = self.tp_size + self.max_pred

    def featurize(self, node):
        if isinstance(node, FilterNode):
            return self.filter_features(node).astype("float64")
        elif isinstance(node, TriplePattern):
            return np.concatenate(
                (self.tp_features(node), self.pred_clust_features(node)), axis=0
            ).astype("float64")
        else:
            raise Exception("unknown node type")

    def pred_clust_features(self, node: TriplePattern):
        vec = np.zeros(self.max_pred)
        try:
            idx = self.pred2index[node.predicate.node_label]
        except KeyError:
            idx = self.max_pred - 1
        vec[idx] = 1
        return vec


class PredStats:
    def __init__(
        self,
        path="/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats",
    ) -> None:
        self.path = path
        self.triple_freq = {}
        self.pred_ents = {}
        self.pred_lits = {}
        self.load_preds_stats()
        # print(len(list(self.triple_freq.keys())))

    def load_preds_freq(self):
        freq_path = self.path + "/freq/"
        if not os.path.exists(freq_path):
            raise Exception("Predicate feature not existing")
        files = sorted(
            [f"{freq_path}{x}" for x in os.listdir(freq_path) if x.endswith(".json")]
        )
        for f in files:
            self.load_pred_freq(f)

    def load_preds_stats(self):
        freq_path = self.path + "/freq/"
        ent_path = self.path + "/ents/"
        lits_path = self.path + "/lits/"
        if not (
            os.path.exists(freq_path)
            and os.path.exists(ent_path)
            and os.path.exists(lits_path)
        ):
            raise Exception("Predicate feature not existing")
        for p, f in zip(
            [freq_path, ent_path, lits_path],
            [self.load_pred_freq, self.load_pred_ents, self.load_pred_lits],
        ):
            self.load_preds_stat_helper(p, f)

    def load_preds_stat_helper(self, path, loader_func):
        files = sorted([f"{path}{x}" for x in os.listdir(path) if x.endswith(".json")])
        for f in files:
            loader_func(f)

    def load_pred_freq(self, file):
        data = json.load(open(file, "r"))
        data = data["results"]["bindings"]
        if not "p1" in data[0].keys():
            return None

        for x in data:
            if x["p1"]["value"] in self.triple_freq.keys():
                assert x["triples"]["value"] == self.triple_freq[x["p1"]["value"]]
            self.triple_freq[x["p1"]["value"]] = x["triples"]["value"]

    def load_pred_ents(self, file):
        data = json.load(open(file, "r"))
        data = data["results"]["bindings"]
        if not "p1" in data[0].keys():
            return None

        for x in data:
            if x["p1"]["value"] in self.pred_ents.keys():
                assert x["entities"]["value"] == self.pred_ents[x["p1"]["value"]]
            self.pred_ents[x["p1"]["value"]] = x["entities"]["value"]

    def load_pred_lits(self, file):
        data = json.load(open(file, "r"))
        data = data["results"]["bindings"]
        if not "p1" in data[0].keys():
            return None

        for x in data:
            if x["p1"]["value"] in self.pred_lits.keys():
                assert x["literals"]["value"] == self.pred_lits[x["p1"]["value"]]
            self.pred_lits[x["p1"]["value"]] = x["literals"]["value"]


class FilterFeatureUtils:
    def isLogical(filter_expr):
        lst = ["||", "&&"]  #'!' can be confused with '!=' comparison operator
        return FilterFeatureUtils.isSubstring(filter_expr, lst)

    def isArithmetic(filter_expr):
        lst = ["*", "+", "/", "-"]
        return FilterFeatureUtils.isSubstring(filter_expr, lst)

    def isComparison(filter_expr):
        lst = ["=", "!=", "<", ">", "<=", ">="]
        return FilterFeatureUtils.isSubstring(filter_expr, lst)

    def isGeneralFunction(filter_expr):
        lst = [
            "DATATYPE",
            "STR",
            "IRI",
            "LANG",
            "BOUND",
            "IN",
            "NOT IN",
            "isBlank",
            "isIRI",
            "isLiteral",
        ]
        return FilterFeatureUtils.isSubstring(filter_expr, lst)

    def isStringManipulation(filter_expr):
        lst = [
            "STRLEN",
            "SUBSTR",
            "UCASE",
            "LCASE",
            "STRSTARTS",
            "STRENDS",
            "CONTAINS",
            "STRBEFORE",
            "STRAFTER",
            "ENCODE_FOR_URI",
            "CONCAT",
            "LANGMATCHES",
            "REGEX",
            "REPLACE",
        ]

        return FilterFeatureUtils.isSubstring(filter_expr, lst)

    def isTime(filter_expr):
        lst = [
            "NOW",
            "YEAR",
            "MONTH",
            "DAY",
            "HOURS",
            "MINUTES",
            "SECONDS",
            "TIMEZONE",
            "TZ",
        ]
        return FilterFeatureUtils.isSubstring(filter_expr, lst)

    def isSubstring(filter_expr, lst):
        filter_expr = filter_expr.lower()
        lst = [x.lower() for x in lst]
        for x in lst:
            if x in filter_expr:
                return True
        return False


# p = PredStats()
# print(len(list(p.pred_ents.keys())))
# print(len(list(p.triple_freq.keys())))
# print(len(list(p.pred_lits.keys())))
