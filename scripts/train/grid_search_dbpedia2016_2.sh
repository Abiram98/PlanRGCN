export TZ=UTC-2
# Get the current date and time
current_datetime=$(date '+%Y_%m_%d_%H_%M')

# Define the list of basedir values
# First run
#configs=("DBpedia2016_sample_0_1_10 0_1_10 3" "DBpedia2016_sample_0_1_10_aug 0_1_10 3" "DBpedia2016_sample_0_1_10_weight_loss 0_1_10 3")
#configs=("DBpedia2016_sample_0_1_10 0_1_10 3" "DBpedia2016_sample_0_1_10_aug 0_1_10 3" "DBpedia2016_sample_0_1_10_weight_loss 0_1_10 3")
#configs=("DBpedia2016_sample_0_300ms_1_10 0_300ms_1_10 4" "DBpedia2016_sample_0_300ms_1_10_aug 0_300ms_1_10 4" "DBpedia2016_sample_0_300ms_1_10_weight_loss 0_300ms_1_10 4")
configs=("DBpedia2016_sample_0_1_10_hybrid 0_1_10 3")
for config in "${configs[@]}"; do
    # Split the config into basedir and snap_lat2onehot_binary
    IFS=' ' read -r basedir snap_value class_num <<< $config
    path_to_output="/PlanRGCN/dbpedia2016_grid/$basedir/$current_datetime"
    mkdir -p $path_to_output
    path_to_base="/qpp/dataset/$basedir"
    python3 /PlanRGCN/trainer/trainer/gridsearch.py \
                --dbpedia2016 \
                --output_path $path_to_output \
                --cls_func $snap_value \
                --n_classes $class_num \
                --base_dir $path_to_base
done
