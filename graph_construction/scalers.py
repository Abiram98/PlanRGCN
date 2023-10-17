from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
import numpy as np


class EntMinMaxScaler:
    def __init__(
        self, ent_freq, ent_subj, ent_obj, pred_freq, pred_ents, pred_lits
    ) -> None:
        # predicate frequency
        self.pred_freq_scaler = MinMaxScaler()
        input_lst = np.array([int(x) for x in pred_freq.values()]).reshape(-1, 1)
        self.pred_freq_scaler.fit(input_lst)

        # Predicate literal counts
        input_lst = np.array([int(x) for x in pred_lits.values()]).reshape(-1, 1)
        self.pred_lits_scaler = MinMaxScaler()
        self.pred_lits_scaler.fit(input_lst)

        # Predicate entity counts
        input_lst = np.array([int(x) for x in pred_ents.values()]).reshape(-1, 1)
        self.pred_ents_scaler = MinMaxScaler()
        self.pred_ents_scaler.fit(input_lst)

        # Entity Frequency
        input_lst = np.array([int(x) for x in ent_freq.values()]).reshape(-1, 1)
        self.ent_freq_scaler = MinMaxScaler()
        self.ent_freq_scaler.fit(input_lst)

        # Entity subj frequency
        input_lst = np.array([int(x) for x in ent_subj.values()]).reshape(-1, 1)
        self.ent_sub_scaler = MinMaxScaler()
        self.ent_sub_scaler.fit(input_lst)

        # Entity Obj frequency
        input_lst = np.array([int(x) for x in ent_obj.values()]).reshape(-1, 1)
        self.ent_obj_scaler = MinMaxScaler()
        self.ent_obj_scaler.fit(input_lst)

    def pred_scale(self, freq, lits, ents):
        #freq = self.pred_freq_scaler.transform([[freq]])[0, 0]
        #lits = self.pred_lits_scaler.transform([[lits]])[0, 0]
        #ents = self.pred_ents_scaler.transform([[ents]])[0, 0]
        return np.log(freq), np.log(lits),np.log( ents)

    def ent_scale(self, ent_freq, subj_freq, obj_freq):
        #ent_freq = self.ent_freq_scaler.transform([[ent_freq]])[0, 0]
        #subj_freq = self.ent_sub_scaler.transform([[subj_freq]])[0, 0]
        #obj_freq = self.ent_obj_scaler.transform([[obj_freq]])[0, 0]
        return np.log(ent_freq), np.log(subj_freq), np.log(obj_freq)


class EntStandardScaler:
    def __init__(
        self, ent_freq, ent_subj, ent_obj, pred_freq, pred_ents, pred_lits
    ) -> None:
        # predicate frequency
        self.pred_freq_scaler = StandardScaler()
        input_lst = np.array([int(x) for x in pred_freq.values()]).reshape(-1, 1)
        self.pred_freq_scaler.fit(input_lst)

        # Predicate literal counts
        input_lst = np.array([int(x) for x in pred_lits.values()]).reshape(-1, 1)
        self.pred_lits_scaler = StandardScaler()
        self.pred_lits_scaler.fit(input_lst)

        # Predicate entity counts
        input_lst = np.array([int(x) for x in pred_ents.values()]).reshape(-1, 1)
        self.pred_ents_scaler = StandardScaler()
        self.pred_ents_scaler.fit(input_lst)

        # Entity Frequency
        input_lst = np.array([int(x) for x in ent_freq.values()]).reshape(-1, 1)
        self.ent_freq_scaler = StandardScaler()
        self.ent_freq_scaler.fit(input_lst)

        # Entity subj frequency
        input_lst = np.array([int(x) for x in ent_subj.values()]).reshape(-1, 1)
        self.ent_sub_scaler = StandardScaler()
        self.ent_sub_scaler.fit(input_lst)

        # Entity Obj frequency
        input_lst = np.array([int(x) for x in ent_obj.values()]).reshape(-1, 1)
        self.ent_obj_scaler = StandardScaler()
        self.ent_obj_scaler.fit(input_lst)

    def pred_scale(self, freq, lits, ents):
        freq = self.pred_freq_scaler.transform([[freq]])[0, 0]
        lits = self.pred_lits_scaler.transform([[lits]])[0, 0]
        ents = self.pred_ents_scaler.transform([[ents]])[0, 0]
        return freq, lits, ents

    def ent_scale(self, ent_freq, subj_freq, obj_freq):
        ent_freq = self.ent_freq_scaler.transform([[ent_freq]])[0, 0]
        subj_freq = self.ent_sub_scaler.transform([[subj_freq]])[0, 0]
        obj_freq = self.ent_obj_scaler.transform([[obj_freq]])[0, 0]
        return ent_freq, subj_freq, obj_freq
