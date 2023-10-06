from sklearn.preprocessing import MinMaxScaler


class EntMinMaxScaler:
    def __init__(
        self, ent_freq, ent_subj, ent_obj, pred_freq, pred_ents, pred_lits
    ) -> None:
        self.pred_freq_scaler = MinMaxScaler()
        self.pred_freq_scaler.fit([int(x) for x in pred_freq.values()])
        self.pred_lits_scaler = MinMaxScaler()
        self.pred_lits_scaler.fit([int(x) for x in pred_lits.values()])
        self.pred_ents_scaler = MinMaxScaler()
        self.pred_ents_scaler.fit([int(x) for x in pred_ents.values()])

        self.ent_freq_scaler = MinMaxScaler()
        self.ent_freq_scaler.fit([int(x) for x in ent_freq.values()])

        self.ent_sub_scaler = MinMaxScaler()
        self.ent_sub_scaler.fit([int(x) for x in ent_subj.values()])
        self.ent_obj_scaler = MinMaxScaler()
        self.ent_obj_scaler.fit([int(x) for x in ent_obj.values()])

    def pred_scale(self, freq, lits, ents):
        freq = self.pred_freq_scaler.transform(freq)
        lits = self.pred_lits_scaler.transform(lits)
        ents = self.pred_ents_scaler.transform(ents)
        return freq, lits, ents

    def ent_scale(self, ent_freq, subj_freq, obj_freq):
        ent_freq = self.ent_freq_scaler(ent_freq)
        subj_freq = self.ent_sub_scaler(subj_freq)
        obj_freq = self.ent_obj_scaler(obj_freq)
        return ent_freq, subj_freq, obj_freq
