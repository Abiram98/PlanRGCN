import json
import os
import numpy as np


class FeaturizerBase:
    def __init__(self, vec_size) -> None:
        self.vec_size = vec_size

    def featurize(self, node):
        return np.array([1, 0, 0, 0, 0])


# empty header definition acutal implementation elsewhere
class FilterNode:
    pass


class TriplePattern:
    pass


class FeaturizerPredCo(FeaturizerBase):
    def __init__(
        self,
        vec_size,
        pred_stat_path="/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats/",
    ) -> None:
        super().__init__(vec_size)
        self.pred_stat_path = pred_stat_path

        self.filter_size = 6

    def featurize(self, node):
        if isinstance(node, FilterNode):
            vec = self.filter_features(node)
        elif isinstance(node, TriplePattern):
            vec = self.tp_features(node)
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
            if f(node):
                vec.append(1)
            else:
                vec.append(0)
        return np.array(vec)

    def tp_features(self, node):
        var_vec = np.array(
            [node.subject.nodetype, node.predicate.nodetype, node.object.nodetype]
        )

        return np.concatenate((var_vec), axis=0)


class PredStats:
    def __init__(
        self,
        path="/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats/",
    ) -> None:
        self.path = path

    def load_preds(self):
        freq_path = self.path + "freq/"
        if not os.path.exists(freq_path):
            raise Exception("Predicate feature not existing")
        files = [
            f"{freq_path}{x}" for x in os.listdir(freq_path) if x.endswith(".json")
        ]

    def load_pred(self, file):
        data = json.load(open(file, "r"))
        data = data["results"]["bindings"]


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
