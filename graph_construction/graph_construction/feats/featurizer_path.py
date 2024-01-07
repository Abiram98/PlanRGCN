import pickle
from graph_construction.feats.feature_binner import FeaturizerBinning
from graph_construction.feats.feat_scale_util import BinnerEntPred
from graph_construction.feats.featurizer import EntStats
from graph_construction.nodes.path_node import PathNode
from graph_construction.qp.qp_utils import pathOpTypes
import numpy as np
from scalers import EntMinMaxScaler, EntStandardScaler
from utils.stats import PredStats


class FeaturizerPath(FeaturizerBinning):
    def __init__(
        self,
        pred_stat_path="/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats",
        pred_com_path="/PlanRGCN/data/pred/pred_co/pred2index_louvain.pickle",
        ent_path="/PlanRGCN/extracted_features/entities/ent_stat/batches_response_stats",
        lit_path = None,
        bins=50,
        pred_end_path=None,
        scaling="None",
    ) -> None:
        self.pred_stat_path = pred_stat_path

        p = PredStats(path=pred_stat_path)
        self.pred_freq = p.triple_freq
        self.pred_ents = p.pred_ents
        self.pred_lits = p.pred_lits

        self.filter_size = 6
        # self.tp_size = 6 # This value will be updated
        # super(FeaturizerPredStats,self).__init__(pred_stat_path)

        self.pred2index, self.max_pred = pickle.load(open(pred_com_path, "rb"))
        estat = EntStats(path=ent_path)
        self.ent_freq = estat.ent_freq
        self.ent_subj = estat.subj_ents
        self.ent_obj = estat.obj_ents
        match scaling:
            case "binner":
                self.scaling = "binner"
                self.scaler = BinnerEntPred(
                    self.ent_freq,
                    self.ent_subj,
                    self.ent_obj,
                    self.pred_freq,
                    self.pred_ents,
                    self.pred_lits,
                    bins=bins,
                )
                self.tp_size = (
                    self.max_pred
                    + 3
                    + self.scaler.ent_scale_len() * 2
                    + self.scaler.pred_scale_len()
                    + 2
                    + pathOpTypes.get_max_operations()
                )
            case "None":
                self.scaling = "binner"
                self.scaler = BinnerEntPred(
                    self.ent_freq,
                    self.ent_subj,
                    self.ent_obj,
                    self.pred_freq,
                    self.pred_ents,
                    self.pred_lits,
                    bins=bins,
                )
            case "std":
                self.scaling = "std"
                self.scaler = EntStandardScaler(
                    self.ent_freq,
                    self.ent_subj,
                    self.ent_obj,
                    self.pred_freq,
                    self.pred_ents,
                    self.pred_lits,
                )
            case "minmax":
                self.scaling = "minmax"
                self.scaler = EntMinMaxScaler(
                    self.ent_freq,
                    self.ent_subj,
                    self.ent_obj,
                    self.pred_freq,
                    self.pred_ents,
                    self.pred_lits,
                )
        self.tp_size = (
            self.max_pred
            + 3
            + self.scaler.ent_scale_len() * 2
            + self.scaler.pred_scale_len()
            + 2
            + pathOpTypes.get_max_operations()
        )

    def tp_features(self, node):
        var_vec = np.array(
            [node.subject.nodetype, node.predicate.nodetype, node.object.nodetype]
        )
        # property path related info
        path_operation = np.zeros(pathOpTypes.get_max_operations())
        pred_min_max = np.zeros(2)
        if isinstance(node, PathNode):
            pred_min_max[0] = node.p_mod_min
            pred_min_max[1] = node.p_mod_max
            for op in node.path_complexity:
                op: pathOpTypes
                path_operation[op.value] = 1

        freq = self.get_value_dict(self.pred_freq, node.predicate.node_label)
        lits = self.get_value_dict(self.pred_lits, node.predicate.node_label)
        ents = self.get_value_dict(self.pred_ents, node.predicate.node_label)

        freq, lits, ents = self.scaler.pred_scale(freq, lits, ents)

        (
            subj_freq,
            subj_subj_freq,
            sub_obj_freq,
            obj_freq,
            obj_subj_freq,
            obj_obj_freq,
        ) = (0, 0, 0, 0, 0, 0)
        if node.subject.type == "URI":
            subj_freq = self.get_value_dict(self.ent_freq, node.subject.node_label)
            subj_subj_freq = self.get_value_dict(self.ent_subj, node.subject.node_label)
            sub_obj_freq = self.get_value_dict(self.ent_obj, node.subject.node_label)
            subj_freq, subj_subj_freq, sub_obj_freq = self.scaler.ent_scale(
                subj_freq, subj_subj_freq, sub_obj_freq
            )
        else:
            subj_freq, subj_subj_freq, sub_obj_freq = self.scaler.ent_scale_no_values()

        if node.object.type == "URI":
            obj_freq = self.get_value_dict(self.ent_freq, node.subject.node_label)
            obj_subj_freq = self.get_value_dict(self.ent_subj, node.subject.node_label)
            obj_obj_freq = self.get_value_dict(self.ent_obj, node.subject.node_label)
            obj_freq, obj_subj_freq, obj_obj_freq = self.scaler.ent_scale(
                obj_freq, obj_subj_freq, obj_obj_freq
            )
        else:
            obj_freq, obj_subj_freq, obj_obj_freq = self.scaler.ent_scale_no_values()

        stat_vec = np.concatenate(
            [
                path_operation,
                pred_min_max,
                freq,
                lits,
                ents,
                subj_freq,
                subj_subj_freq,
                sub_obj_freq,
                obj_freq,
                obj_subj_freq,
                obj_obj_freq,
            ]
        )

        return np.concatenate((var_vec, stat_vec, np.zeros(self.filter_size)), axis=0)
