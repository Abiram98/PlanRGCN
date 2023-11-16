from graph_construction.node import TriplePattern, is_variable_check
from graph_construction.node import Node
from graph_construction.qp.qp_utils import pathOpTypes


class PathNode(TriplePattern):
    def __init__(self, data: dict, node_class=Node):
        self.depthLevel = None
        self.node_class = node_class
        # Not implemented for now
        # self.path_predicates = list()
        # for p in data["Predicates"]:
        #    self.path_predicates.append(node_class(p))

        self.subject = node_class(data["Subject"])
        predicate = data["Predicates"][0]
        if isinstance(predicate, str):
            self.predicate = node_class(predicate)
            self.p_mod_max = 0
            self.p_mod_min = 0
        else:
            self.predicate = node_class(predicate["Predicate"])
            self.p_mod_max = predicate["min"]
            self.p_mod_min = predicate["max"]
        self.path_complexity: list[pathOpTypes] = list()
        for comp in data["pathComplexity"]:
            self.path_complexity.append(pathOpTypes.get_path_op(comp))

        self.object = node_class(data["Object"])

        # with good results of 80% f1 score - old encoding - Not tested with path though
        self.subject.nodetype = 0
        self.predicate.nodetype = 1
        self.object.nodetype = 2

        # New variable encoding
        self.subject.nodetype = 0 if is_variable_check(self.subject.node_label) else 1
        self.predicate.nodetype = (
            0 if is_variable_check(self.predicate.node_label) else 1
        )
        self.object.nodetype = 0 if is_variable_check(self.object.node_label) else 1
        if "level" in data.keys():
            self.level = data["level"]
            self.depthLevel = self.level

    def __str__(self):
        return f"PATH ({str(self.subject)} {str(self.predicate)} {str(self.object)} )"

    def __repr__(self):
        return f"PATH ({str(self.subject)} {str(self.predicate)} {str(self.object)} )"
