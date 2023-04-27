

import configparser

import torch
from feature_extraction.constants import PATH_TO_CONFIG_GRAPH
from feature_extraction.predicate_features_sub_obj import Predicate_Featurizer_Sub_Obj
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
#Corresponding predicate feature class for node_num_pred.py class
class Pred_Feat_Num_Enc(Predicate_Featurizer_Sub_Obj):
    def __init__(self, endpoint_url=None, timeout=30):
        super().__init__(endpoint_url, timeout)
    
    #this function will be invoked during the loading of the function.
    def prepare_pred_feat(self, bins = 30, k=20):
        def min_max_scaling(val, min, max):
            return (val-min)/(max-min)
        
        dct = {'predicate':[], 'freq':[]}
        for key in self.predicate_freq.keys():
            dct['predicate'].append(key)
            dct['freq'].append(self.predicate_freq[key])
        df = pd.DataFrame.from_dict(dct)
        #df_freq = df['freq'].astype('int')
        df['freq'] = pd.to_numeric(df['freq'])
        df_freq = df.sort_values('freq',ascending=False)
        df = df_freq
        df_freq = df_freq.assign(row_number=range(len(df_freq)))
        df_norm = df_freq
        df_freq = df_freq.set_index('predicate')
        
        self.preq_min = df_norm.freq.min()
        self.preq_max = df_norm.freq.max()
        df_norm['norm_freq'] = df.freq.apply(min_max_scaling, args=(self.preq_min, self.preq_max,))
        df_norm = df_norm.set_index('predicate')
        self.norm_df = df_norm
        
        self.pred_encoder = df_freq
        self.no_preds = len(df_freq)
    
    def get_pred_feat(self, pred_label):
        return self.pred_encoder.loc[pred_label]['row_number']

    #Min max scaling is not working
    def get_pred_feat2(self,pred_label):
        try:
            return self.norm_df.loc[pred_label]['norm_freq']
        except KeyError:
            return -1
    
    

    
    #TOdo define functions to be used 

if __name__ == "__main__":
    #Testing
    parser = configparser.ConfigParser()
    parser.read(PATH_TO_CONFIG_GRAPH)
    feat_generation_path = parser['PredicateFeaturizerSubObj']['load_path']
    topk = int(parser['PredicateFeaturizerSubObj']['topk'])
    bin_no = int(parser['PredicateFeaturizerSubObj']['bin_no'])
    pred_feature_rizer:Pred_Feat_Num_Enc = Pred_Feat_Num_Enc.prepare_pred_featues_for_bgp(feat_generation_path, bins=bin_no, topk=topk,obj_type=Pred_Feat_Num_Enc)
    print(len(pred_feature_rizer.predicate_freq.keys()))
    #for key in pred_feature_rizer.predicate_freq.keys():
    #    print(key)
    #    print(pred_feature_rizer.get_pred_feat(key))