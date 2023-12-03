from graph_construction.feats.feature_binner import FeaturizerBinning
from graph_construction.feats.featurizer_path import FeaturizerPath
from trainer.train_ray import main
from graph_construction.feats.featurizer import FeaturizerPredCoEnt
from graph_construction.query_graph import (
    QueryPlan,
    QueryPlanCommonBi,
    snap_lat2onehot,
    snap_lat2onehotv2,
)
from graph_construction.qp.query_plan_path import QueryPlanPath
from ray import tune
import os


sample_name = "wikidata_0_1_10_v2_aug"
sample_name = "wikidata_0_1_10_v2_weight_loss"
sample_name = "wikidata_0_1_10_v2_hybrid"
sample_name = "wikidata_0_1_10_v2_path_aug"
sample_name = "wikidata_0_1_10_v2_path"  # balanced dataset
sample_name = "wikidata_0_1_10_v2_path_weight_loss"
sample_name = "wikidata_0_1_10_v2_path_hybrid"

# Results save path
"""if os.path.exists(path_to_save):
    resume = True
else:
    resume=False"""

# Dataset split paths
#train_path = f"/qpp/dataset/{sample_name}/train_sampled.tsv"
#val_path = f"/qpp/dataset/{sample_name}/val_sampled.tsv"
#test_path = f"/qpp/dataset/{sample_name}/test_sampled.tsv"

train_path = f"/data/{sample_name}/train_sampled.tsv"
val_path = f"/data/{sample_name}/val_sampled.tsv"
test_path = f"/data/{sample_name}/test_sampled.tsv"
qp_path = f"/data/{sample_name}/queryplans/"

# KG statistics feature paths
pred_stat_path = (
    "/data/extracted_features_wd/predicate/pred_stat/batches_response_stats"
)
pred_com_path = "/data/extracted_features_wd/predicate/pred_co"
#ent_path = (
#    "/PlanRGCN/extracted_features_dbpedia2016/entities/ent_stat/batches_response_stats"
#)

ent_path = (
    "/data/extracted_features_wd/entities/ent_stat/batches_response_stats"
)
# Training Configurations
num_samples = 1  # cpu cores to use
num_samples = 8  # cpu cores to use
num_samples = 1  # cpu cores to use
max_num_epochs = 100
# batch_size = 64
query_plan_dir = qp_path
time_col = "mean_latency"
is_lsq = True
cls_func = snap_lat2onehotv2
featurizer_class = FeaturizerPredCoEnt
featurizer_class = FeaturizerBinning
featurizer_class = FeaturizerPath
#scaling = "robust"
n_classes = 3
query_plan = QueryPlan
query_plan = QueryPlanPath
scaling = "minmax"
scaling = "binner"
prepper = None
resume = False
path_to_save = f"/data/{sample_name}/planrgcn_{scaling}"
os.makedirs(path_to_save, exist_ok=True)
config = {
    "l1": tune.grid_search([10]),
    "l2": tune.grid_search([10]),
    "dropout": tune.grid_search([0.0]),
    "wd": 0.01,
    "lr": tune.grid_search([1e-5]),
    "epochs": 1,
    "batch_size": tune.grid_search([64]),
    "loss_type": "cross-entropy",
}
config = {
    "l1": tune.choice([64,  512, 1024, 2048, 4096]),
    "l2": tune.choice([64, 512, 1024, 2048, 4096]),
    "dropout": tune.choice([0.0, 0.6]),
    "wd": 0.01,
    "lr": tune.grid_search([1e-5]),
    "epochs": 100,
    "batch_size": tune.choice([128, 256]),
    "loss_type": tune.choice(["cross-entropy","mse"]),
    "pred_com_path": tune.choice(
        [ "pred2index_louvain.pickle"]
    ),
}
main(
    num_samples=num_samples,
    max_num_epochs=max_num_epochs,
    train_path=train_path,
    val_path=val_path,
    test_path=test_path,
    query_plan_dir=qp_path,
    pred_stat_path=pred_stat_path,
    pred_com_path=pred_com_path,
    ent_path=ent_path,
    time_col=time_col,
    is_lsq=is_lsq,
    cls_func=cls_func,
    featurizer_class=featurizer_class,
    scaling=scaling,
    n_classes=n_classes,
    query_plan=query_plan,
    path_to_save=path_to_save,
    config=config,
    resume = resume
)