echo "Training model in QPP2"


EXP_NAME="Wikidata Training"
FEAT=/data/metaKGStat/wikidata
EXP=/data/wikidata_3_class_full/plan_something
BASEFOLDERNAME=wikidata_3_class_full
LAYER1=8192
LAYER2=4096
NCLASSES=3
mkdir -p $EXP
echo "Train Start" $EXP_NAME $SECONDS >> $EXP/train_log.txt
python3 /PlanRGCN/scripts/train/ray_run.py $BASEFOLDERNAME $EXP --feat_path $FEAT --use_pred_co no --layer1_size $LAYER1 --layer2_size $LAYER2
echo "Train DONE" $EXP_NAME $SECONDS >> $EXP/train_log.txt
python3 -m trainer.predict2 -p "$EXP"/prepper.pcl -m "$EXP"/best_model.pt -n $NCLASSES -o "$EXP" --l1 $LAYER1 --l2 $LAYER2
echo "Prediction DONE" $EXP_NAME $SECONDS >> $EXP/train_log.txt






exit

## Temp code for making new splits
F=/data/wikidataV2
OF=/data/wikidata_3_class_full
mkdir $F
cd $F
cp $OF/all.tsv $F
cp $OF/test_sampled.tsv $F
cp $OF/train_sampled.tsv $F
cp $OF/val_sampled.tsv $F