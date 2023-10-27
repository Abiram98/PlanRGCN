export TZ=UTC-2
# Get the current date and time
current_datetime=$(date '+%Y_%m_%d_%H_%M')

# Define the list of basedir values
configs=("wikidata_0_1_10 0_1_10 3" "wikidata_0_1_10_weight_loss 0_1_10 3" "wikidata_0_1_10_aug 0_1_10 3")
for config in "${configs[@]}"; do
    # Split the config into basedir and snap_lat2onehot_binary
    IFS=' ' read -r basedir snap_value class_num <<< $config
    path_to_output="/PlanRGCN/wikidata/$basedir/$current_datetime"
    mkdir -p $path_to_output
    path_to_base="/qpp/dataset/$basedir"
    python3 /PlanRGCN/trainer/trainer/gridsearch.py \
                --dbpedia2016 \
                --output_path $path_to_output \
                --cls_func $snap_value \
                --n_classes $class_num \
                --base_dir $path_to_base
done