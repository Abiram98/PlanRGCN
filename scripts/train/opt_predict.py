
from graph_construction.query_graph import snap_lat2onehotv2
from trainer.train_ray import get_dataloaders
import sys
from graph_construction.feats.featurizer_path import FeaturizerPath
from graph_construction.qp.query_plan_path import QueryPlanPath
from ray import tune
import os

sample_name = sys.argv[1]
path_to_save = sys.argv[2]
use_pred_co = sys.argv[4]
train_path = f"/data/{sample_name}/train_sampled.tsv"
val_path = f"/data/{sample_name}/val_sampled.tsv"
test_path = f"/data/{sample_name}/test_sampled.tsv"
qp_path = f"/data/{sample_name}/queryplans/"
save_prep_path=f'{path_to_save}/prepper.pcl'
feat_base_path='/data/planrgcn_feat/extracted_features_dbpedia2016'
feat_base_path=sys.argv[3]

# KG statistics feature paths
pred_stat_path = (
    f"{feat_base_path}/predicate/pred_stat/batches_response_stats"
)

if use_pred_co == 'no':
    pred_com_path = None
else:
    pred_com_path = f"{feat_base_path}/predicate/pred_co"

ent_path = (
    f"{feat_base_path}/entity/ent_stat/batches_response_stats"
)

lit_path= (
    f"{feat_base_path}/literals/literals_stat/batches_response_stats"
)

# Training Configurations
num_samples = 22  # 4
num_cpus= 22
max_num_epochs = 100
query_plan_dir = qp_path
time_col = "mean_latency"
is_lsq = True
cls_func = snap_lat2onehotv2
#featurizer_class = FeaturizerBinning
featurizer_class = FeaturizerPath
scaling = "binner"
n_classes = 3
query_plan = QueryPlanPath
prepper = None
resume=False

val_pp_path= None,
time_col="mean_latency",
config = {
    "l1": tune.choice([ 1024, 2048, 4096]),
    "l2": tune.choice([ 512, 1024, 2048, 4096]),
    "dropout": tune.choice([0.0, 0.6]),
    "wd": 0.01,
    "lr": tune.grid_search([1e-5]),
    "epochs": 100,
    "batch_size": tune.choice([ 256]),
    "loss_type": tune.choice(["cross-entropy"]),
    "pred_com_path": tune.choice(
        [ "pred2index_louvain.pickle"]
    ),
}

resume=False
patience=5
    
if pred_com_path == None:
    con_pred_com_path = None
else:
    con_pred_com_path = os.path.join(pred_com_path, config["pred_com_path"])

train_loader, val_loader, test_loader, input_d, val_pp_loader = get_dataloaders(
    train_path=train_path,
    val_path=val_path,
    test_path=test_path,
    val_pp_path= val_pp_path,
    batch_size=config["batch_size"],
    query_plan_dir=query_plan_dir,
    pred_stat_path=pred_stat_path,
    pred_com_path=con_pred_com_path,
    ent_path=ent_path,
    lit_path=lit_path,
    time_col=time_col,
    is_lsq=is_lsq,
    cls_func=cls_func,
    featurizer_class=featurizer_class,
    scaling=scaling,
    query_plan=query_plan,
    save_prep_path=save_prep_path,
    save_path= None,
    config=config
)
