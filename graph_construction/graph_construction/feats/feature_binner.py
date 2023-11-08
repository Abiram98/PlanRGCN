from graph_construction.feats.featurizer import (
    EntStats,
    FeaturizerPredCoEnt,
    FeaturizerPredStats,
)
import pickle
from graph_construction.node import FilterNode, TriplePattern
from sklearn.preprocessing import KBinsDiscretizer
import numpy as np
from utils.stats import PredStats


class BinnerEntPred:
    def __init__(
        self,
        ent_freq,
        ent_subj,
        ent_obj,
        pred_freq,
        pred_ents,
        pred_lits,
        bins=50,
        random_state=42,
    ) -> None:
        # predicate frequency
        KBinsDiscretizer(n_bins=50, strategy="quantile", encode="ordinal")
        self.pred_freq_scaler = KBinsDiscretizer(
            n_bins=bins,
            strategy="quantile",
            encode="onehot-dense",
            random_state=random_state,
        )
        input_lst = np.array(sorted([int(x) for x in pred_freq.values()])).reshape(
            -1, 1
        )
        self.pred_freq_scaler.fit(input_lst)

        # Predicate literal counts
        input_lst = np.array(sorted([int(x) for x in pred_lits.values()])).reshape(
            -1, 1
        )
        self.pred_lits_scaler = KBinsDiscretizer(
            n_bins=bins,
            strategy="quantile",
            encode="onehot-dense",
            random_state=random_state,
        )
        self.pred_lits_scaler.fit(input_lst)

        # Predicate entity counts
        input_lst = np.array(sorted([int(x) for x in pred_ents.values()])).reshape(
            -1, 1
        )
        self.pred_ents_scaler = KBinsDiscretizer(
            n_bins=bins,
            strategy="quantile",
            encode="onehot-dense",
            random_state=random_state,
        )
        self.pred_ents_scaler.fit(input_lst)

        # Entity Frequency
        input_lst = np.array(sorted([int(x) for x in ent_freq.values()])).reshape(-1, 1)
        self.ent_freq_scaler = KBinsDiscretizer(
            n_bins=bins,
            strategy="quantile",
            encode="onehot-dense",
            random_state=random_state,
        )
        self.ent_freq_scaler.fit(input_lst)

        # Entity subj frequency
        input_lst = np.array(sorted([int(x) for x in ent_subj.values()])).reshape(-1, 1)
        self.ent_sub_scaler = KBinsDiscretizer(
            n_bins=bins,
            strategy="quantile",
            encode="onehot-dense",
            random_state=random_state,
        )
        self.ent_sub_scaler.fit(input_lst)

        # Entity Obj frequency
        input_lst = np.array(sorted([int(x) for x in ent_obj.values()])).reshape(-1, 1)
        self.ent_obj_scaler = KBinsDiscretizer(
            n_bins=bins,
            strategy="quantile",
            encode="onehot-dense",
            random_state=random_state,
        )
        self.ent_obj_scaler.fit(input_lst)

    def pred_scale(self, freq, lits, ents):
        freq = self.pred_freq_scaler.transform([[freq]])[0]
        lits = self.pred_lits_scaler.transform([[lits]])[0]
        ents = self.pred_ents_scaler.transform([[ents]])[0]
        # return np.log(freq), np.log(lits), np.log(ents)
        return freq, lits, ents

    def pred_scale_len(self):
        return (
            self.pred_freq_scaler.n_bins_[0]
            + self.pred_lits_scaler.n_bins_[0]
            + self.pred_ents_scaler.n_bins_[0]
        )

    def ent_scale(self, ent_freq, subj_freq, obj_freq):
        ent_freq = self.ent_freq_scaler.transform([[ent_freq]])[0]
        subj_freq = self.ent_sub_scaler.transform([[subj_freq]])[0]
        obj_freq = self.ent_obj_scaler.transform([[obj_freq]])[0]
        return ent_freq, subj_freq, obj_freq

    def ent_scale_len(self):
        return (
            self.ent_freq_scaler.n_bins_[0]
            + self.ent_sub_scaler.n_bins_[0]
            + self.ent_obj_scaler.n_bins_[0]
        )

    def ent_scale_no_values(self):
        ent_freq = np.zeros(self.ent_freq_scaler.n_bins_[0])
        subj_freq = np.zeros(self.ent_sub_scaler.n_bins_[0])
        obj_freq = np.zeros(self.ent_obj_scaler.n_bins_[0])
        return ent_freq, subj_freq, obj_freq


class FeaturizerBinning(FeaturizerPredStats):
    def __init__(
        self,
        pred_stat_path="/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats",
        pred_com_path="/PlanRGCN/data/pred/pred_co/pred2index_louvain.pickle",
        ent_path="/PlanRGCN/extracted_features/entities/ent_stat/batches_response_stats",
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
        self.tp_size = 6
        # super(FeaturizerPredStats,self).__init__(pred_stat_path)

        self.pred2index, self.max_pred = pickle.load(open(pred_com_path, "rb"))
        estat = EntStats(path=ent_path)
        self.ent_freq = estat.ent_freq
        self.ent_subj = estat.subj_ents
        self.ent_obj = estat.obj_ents

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
        )

    def featurize(self, node):
        if isinstance(node, FilterNode):
            return self.filter_features(node).astype("float32")
        elif isinstance(node, TriplePattern):
            return np.concatenate(
                (self.tp_features(node), self.pred_clust_features(node)), axis=0
            ).astype("float32")
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

    def tp_features(self, node):
        var_vec = np.array(
            [node.subject.nodetype, node.predicate.nodetype, node.object.nodetype]
        )
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
