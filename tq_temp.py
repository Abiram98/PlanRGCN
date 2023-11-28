from trainer.train import Trainer
from graph_construction.feats.featurizer import FeaturizerPredCo
from graph_construction.query_graph import QueryPlan, snap_lat2onehot, snap_lat2onehotv2
from trainer.model import Classifier as CLS

t = Trainer(
    featurizer_class=FeaturizerPredCo,
    train_path="/qpp/dataset/DBpedia_2016_12k_simple_opt_filt/train_sampled.tsv",
    val_path="/qpp/dataset/DBpedia_2016_12k_simple_opt_filt/val_sampled.tsv",
    test_path="/qpp/dataset/DBpedia_2016_12k_simple_opt_filt/test_sampled.tsv",
    batch_size=32,
    query_plan_dir="/PlanRGCN/extracted_features/queryplans/",
    pred_stat_path="/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats",
    time_col="mean_latency",
    hidden_dim=48,
    n_classes=3,
    query_plan=QueryPlan,
    model=CLS,
    cls_func=snap_lat2onehotv2,
)
t.train(
    epochs=100,
    verbosity=2,
    result_path="/PlanRGCN/results/results_new.json",
    path_to_save="/PlanRGCN/plan_model_new",
)
t.predict(path_to_save="/PlanRGCN/results_new")

exit()
from graph_construction.query_graph import QueryPlan
import trainer.data_util as u

train_path = "/qpp/dataset/DBpedia_2016_12k_simple_opt_filt/train_sampled.tsv"
val_path = "/qpp/dataset/DBpedia_2016_12k_simple_opt_filt/val_sampled.tsv"
test_path = "/qpp/dataset/DBpedia_2016_12k_simple_opt_filt/test_sampled.tsv"

d = u.DatasetPrep(
    train_path=train_path, test_path=test_path, val_path=val_path, query_plan=QueryPlan
)
print(d.get_testloader())
