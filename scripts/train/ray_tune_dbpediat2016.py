train_path = "/qpp/dataset/DBpedia2016_sample_0_1_10/train_sampled.tsv"
val_path = "/qpp/dataset/DBpedia2016_sample_0_1_10/val_sampled.tsv"
test_path = "/qpp/dataset/DBpedia2016_sample_0_1_10/test_sampled.tsv"
qp_path = "/qpp/dataset/DBpedia2016_sample_0_1_10/queryplans/"

num_samples = 2  # cpu cores to use
max_num_epochs = 100
batch_size = 64
query_plan_dir = qp_path
pred_stat_path = "/PlanRGCN/extracted_features_dbpedia2016/predicate/pred_stat/batches_response_stats"
pred_com_path = "/PlanRGCN/extracted_features_dbpedia2016/predicate/pred_co/pred2index_louvain.pickle"
ent_path = (
    "/PlanRGCN/extracted_features_dbpedia2016/entities/ent_stat/batches_response_stats"
)
time_col = "mean_latency"
is_lsq = True
cls_func = snap_lat2onehotv2
featurizer_class = FeaturizerPredCoEnt
scaling = "std"
n_classes = 3
query_plan = QueryPlan
prepper = None
