

pred_com_dir=/PlanRGCN/extracted_features_wd/pred_co_graph
mkdir -p $pred_com_dir
pred_com_path="$pred_com_dir"/pred2index_louvain.pickle
co_pred_path=/PlanRGCN/extracted_features_wd/predicate/pred_co/
# Predicate community constuction step:
: '
python3 -c """
from feat_rep.pred.pred_co import PredicateCommunityCreator,create_louvain_to_p_index
d = PredicateCommunityCreator(save_dir='$pred_com_dir')
d.get_louvain_communities(
    dir='$co_pred_path'
)
create_louvain_to_p_index(path='$pred_com_dir/communities_louvain.pickle',
output_path='$pred_com_path')
"""
'

path_wdbench_alg=/SPARQLBench/wdbench/bgp_opts.tsv
queryplandir=/PlanRGCN/extracted_features_wd/queryplans/
path_to_models=/PlanRGCN/wdbench/models
path_to_res=/PlanRGCN/wdbench/results
mkdir -p $path_to_models
mkdir -p $path_to_res
rm $path_to_models/*
rm $path_to_res/*
split_dir=/qpp/dataset/wdbench
batch_size=8
pred_stat_path=/PlanRGCN/extracted_features_wd/predicate/pred_stat/batches_response_stats
neurons=256

ent_path=/PlanRGCN/extracted_features_wd/entities/ent_stat/batches_response_stats
scaling=\"None\"

python3 -c """
from trainer.train import Trainer
from graph_construction.query_graph import QueryPlan

import os
import dgl
from graph_construction.featurizer import FeaturizerPredCoEnt
from graph_construction.query_graph import QueryPlanCommonBi, snap_lat2onehotv2
from trainer.data_util import DatasetPrep
from trainer.model import Classifier as CLS
import torch as th
import numpy as np
import json
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
import torch.nn.functional as F
import pandas as pd

t = Trainer(
    train_path='$split_dir/train_sampled.tsv',
    val_path='$split_dir/val_sampled.tsv',
    test_path='$split_dir/test_sampled.tsv',
    batch_size=$batch_size,
    query_plan_dir='$queryplandir',
    pred_stat_path='$pred_stat_path',
    pred_com_path='$pred_com_path',
    ent_path='$ent_path',
    time_col='mean_latency',
    cls_func=snap_lat2onehotv2,
    # in_dim=12,
    hidden_dim=$neurons,
    n_classes=3,
    featurizer_class=FeaturizerPredCoEnt,
    scaling=$scaling,
    query_plan=QueryPlan,
    is_lsq=False,
    model=CLS
)

t.train(epochs=100,verbosity=2,
result_path='"$path_to_res"/results.json',
path_to_save='$path_to_models')
t.predict(path_to_save='$path_to_res')
"""

