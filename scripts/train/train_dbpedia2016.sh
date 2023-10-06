queryplandir=/PlanRGCN/extracted_features/queryplans/
path_to_models=/PlanRGCN/dbpedia2016/models3
path_to_res=/PlanRGCN/dbpedia2016/results3
mkdir -p $path_to_models
mkdir -p $path_to_res
rm $path_to_models/*
rm $path_to_res/*
split_dir=/qpp/dataset/DBpedia_2016_12k_simple_opt_filt
#split_dir=/qpp/dataset/DBpedia_2016_12k_sample_simple/
batch_size=32
pred_stat_path=/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats
pred_com_path=/PlanRGCN/data/pred/pred_co/pred2index_louvain.pickle

neurons=256

ent_path=/PlanRGCN/extracted_features/entities/ent_stat/batches_response_stats

earlystop=20
scaling=\"None\"

python3 -c """
from trainer.train import Trainer
from graph_construction.featurizer import FeaturizerPredCoEnt
from graph_construction.query_graph import QueryPlan, snap_lat2onehot,snap_lat2onehotv2,snap_lat2onehot_binary
from trainer.model import Classifier as CLS
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
    cls_func=snap_lat2onehot_binary,
    # in_dim=12,
    hidden_dim=$neurons,
    n_classes=2,
    featurizer_class=FeaturizerPredCoEnt,
    scaling=$scaling,
    query_plan=QueryPlan,
    is_lsq=True,
    model=CLS
)
t.train(epochs=100,verbosity=2,
result_path='"$path_to_res"/results.json',
path_to_save='$path_to_models',
early_stop=$earlystop)
t.predict(path_to_save='$path_to_res')
"""
