export TZ=UTC-2
# Get the current date and time
current_datetime=$(date '+%Y_%m_%d_%H_%M')

: '
export queryplandir="${queryplandir:-/PlanRGCN/extracted_features/queryplans/}"
export path_to_models="${path_to_models:-/PlanRGCN/dbpedia2016/models3}"
export path_to_res="${path_to_res:-/PlanRGCN/dbpedia2016/results3}"
export split_dir="${split_dir:-/qpp/dataset/DBpedia_2016_12k_simple_opt_filt}"
export batch_size="${batch_size:-32}"
export pred_stat_path="${pred_stat_path:-/PlanRGCN/extracted_features/predicate/pred_stat/batches_response_stats}"
export pred_com_path="${pred_com_path:-/PlanRGCN/extracted_features_dbpedia2016/predicate/pred_co/pred2index_louvain.pickle}"
export neurons="${neurons:-256}"
export ent_path="${ent_path:-/PlanRGCN/extracted_features/entities/ent_stat/batches_response_stats}"
export earlystop="${earlystop:-10}"
export scaling="${scaling:-\"std\"}"
'

batch_size=32
pred_stat_path=/PlanRGCN/extracted_features_dbpedia2016/predicate/pred_stat/batches_response_stats
pred_com_path=/PlanRGCN/extracted_features_dbpedia2016/predicate/pred_co/pred2index_louvain.pickle
neurons=256
ent_path=/PlanRGCN/extracted_features_dbpedia2016/entities/ent_stat/batches_response_stats
earlystop=10
scaling=\"std\"

: '
basedir=DBpedia2016_sample_0_1
queryplandir=/qpp/dataset/"$basedir"/queryplans
path_to_models=/PlanRGCN/dbpedia2016/"$basedir"/models
path_to_res=/PlanRGCN/dbpedia2016/"$basedir"/results
split_dir=/qpp/dataset/"$basedir"
mkdir -p $path_to_models
mkdir -p $path_to_res
rm $path_to_models/*
rm $path_to_res/*

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
'
# Define the list of basedir values
basedirs=("DBpedia2016_sample_0_1 2" "DBpedia2016_sample_0_1_10 3")
configs=("DBpedia2016_sample_0_1 snap_lat2onehot_binary 2" "DBpedia2016_sample_0_1_10 snap_lat2onehotv2 3")
for config in "${configs[@]}"; do
    # Split the config into basedir and snap_lat2onehot_binary
    IFS=' ' read -r basedir snap_value class_num <<< $config

    queryplandir="/qpp/dataset/$basedir/queryplans"
    split_dir="/qpp/dataset/$basedir"
    path_to_models="/PlanRGCN/dbpedia2016/$basedir/$current_datetime/models"
    path_to_res="/PlanRGCN/dbpedia2016/$basedir/$current_datetime/results"

    # Create directories if they don't exist
    mkdir -p "$path_to_models"
    mkdir -p "$path_to_res"

    # Clear existing content in directories
    rm -f "$path_to_models"/*
    rm -f "$path_to_res"/*

    # Run the Python script
    python3 -c """
from trainer.train import Trainer
from graph_construction.featurizer import FeaturizerPredCoEnt
from graph_construction.query_graph import QueryPlan, snap_lat2onehot, snap_lat2onehotv2, snap_lat2onehot_binary
from trainer.model import Classifier as CLS

# Create a Trainer instance
t = Trainer(
    train_path='$split_dir/train_sampled.tsv',
    val_path='$split_dir/val_sampled.tsv',
    test_path='$split_dir/test_sampled.tsv',
    batch_size=$batch_size,
    query_plan_dir='$queryplandir/',
    pred_stat_path='$pred_stat_path',
    pred_com_path='$pred_com_path',
    ent_path='$ent_path',
    time_col='mean_latency',
    cls_func="$snap_value",
    hidden_dim=$neurons,
    n_classes=$class_num,
    featurizer_class=FeaturizerPredCoEnt,
    scaling=$scaling,
    query_plan=QueryPlan,
    is_lsq=True,
    model=CLS
)

# Train the model
t.train(epochs=100, verbosity=2,
        result_path='"$path_to_res"/results.json',
        path_to_save='$path_to_models',
        early_stop=$earlystop,
        loss_type='cross-entropy')

# Make predictions
t.predict(path_to_save='$path_to_res')
    """
exit
done
