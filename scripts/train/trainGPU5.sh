
#hyper parameter tune DBpediaV2
source /virt_env_develop/bin/activate
EXP_NAME="Hyperparam search DBpediaV2"
FEAT=/data/metaKGStat/dbpedia
DATESTRING=18_10_2024
EXP=/data/DBpediaV2/plan${DATESTRING}
mkdir -p $EXP
echo "Train Start" $EXP_NAME $SECONDS >> $EXP/hyper_log.txt
python3 /PlanRGCN/scripts/train/ray_hyperparam.py DBpediaV2 $EXP --feat_path $FEAT --use_pred_co no
echo "Train DONE" $EXP_NAME $SECONDS >> $EXP/hyper_log.txt
#python3 -m trainer.predict2 -p "$EXP"/prepper.pcl -m "$EXP"/best_model.pt -n 3 -o "$EXP" --l1 $L1 --l2 $L2
#echo "Prediction DONE" $EXP_NAME $SECONDS >> $EXP/train_log.txt


exit
source /virt_env_develop/bin/activate
EXP_NAME="DBpedia Training"
FEAT=/data/metaKGStat/dbpedia
L1=4096
L2=2048
DATESTRING=16_10_2024
EXP=/data/DBpedia_3_class_full/plan${DATESTRING}_${L1}_${L2}
mkdir -p $EXP
echo "Train Start" $EXP_NAME $SECONDS >> $EXP/train_log.txt
python3 /PlanRGCN/scripts/train/ray_run.py DBpedia_3_class_full $EXP --feat_path $FEAT --use_pred_co no --layer1_size $L1 --layer2_size $L2
echo "Train DONE" $EXP_NAME $SECONDS >> $EXP/train_log.txt
python3 -m trainer.predict2 -p "$EXP"/prepper.pcl -m "$EXP"/best_model.pt -n 3 -o "$EXP" --l1 $L1 --l2 $L2
echo "Prediction DONE" $EXP_NAME $SECONDS >> $EXP/train_log.txt



exit
source /virt_env_develop/bin/activate
EXP_NAME="DBpedia Training"
FEAT=/data/metaKGStat/dbpedia
EXP=/data/DBpedia_3_class_full/plan16_10_2024
mkdir -p $EXP
echo "Train Start" $EXP_NAME $SECONDS >> $EXP/train_log.txt
python3 /PlanRGCN/scripts/train/ray_run.py DBpedia_3_class_full $EXP --feat_path $FEAT --use_pred_co no --layer1_size 4096 --layer2_size 1024
echo "Train DONE" $EXP_NAME $SECONDS >> $EXP/train_log.txt
python3 -m trainer.predict2 -p "$EXP"/prepper.pcl -m "$EXP"/best_model.pt -n 3 -o "$EXP" --l1 4096 --l2 1024
echo "Prediction DONE" $EXP_NAME $SECONDS >> $EXP/train_log.txt



exit
EXP_NAME="Debug_dataset"
FEAT=/data/metaKGStat/dbpedia
EXP=/data/test_set/plan16_10_2024
mkdir -p $EXP
echo "Train Start" $EXP_NAME $SECONDS >> $EXP/train_log.txt
python3 /PlanRGCN/scripts/train/ray_run.py test_set $EXP --feat_path $FEAT --use_pred_co no --layer1_size 4096 --layer2_size 1024
echo "Train DONE" $EXP_NAME $SECONDS >> $EXP/train_log.txt
python3 -m trainer.predict2 -p "$EXP"/prepper.pcl -m "$EXP"/best_model.pt -n 3 -o "$EXP" --l1 4096 --l2 1024
echo "Prediction DONE" $EXP_NAME $SECONDS >> $EXP/train_log.txt