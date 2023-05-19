#!bin/bash
if [ $1 = "strat-split" ]
then
    echo "Strat split"
    mkdir -p /work/data/splits
    (cd ../ && python3 -m preprocessing.utils_script stratified_split \
    --input /work/data/data_files/processed_gt.json \
    --output /work/data/splits )
elif [ $1 = "gt-assign" ]
then
    (cd ../ && python3 -m preprocessing.utils_script gt_assign \
    --input /work/data/data_files/converted_all_data.json \
    --output /work/data/data_files/processed_gt.json \
    --gt_type avg_re)
    #--gt_type std)
    #--gt_type re)
elif [ $1 = "triple-stat" ]
then
    (cd ../ && python3 -m preprocessing.utils_script triple_stat \
    --input /work/data/data_files/processed_gt.json)
elif [ $1 = "convert-leaf" ]
then
    (cd ../ && python3 -m preprocessing.utils_script convert_leaf \
    --input /work/data/data_files/all_data.json \
    --output /work/data/data_files/converted_all_data.json)
    echo "Done converting Leapfrog data to usable format"
elif [ $1 = "latency-stat" ]
then
    (cd ../ && python3 -m preprocessing.utils_script lat_stat \
    --input /work/data/data_files/processed_gt.json )
    echo "Done printing latency stats"
fi
#(cd ../ && python3 -m preprocessing.dataset_analysis)