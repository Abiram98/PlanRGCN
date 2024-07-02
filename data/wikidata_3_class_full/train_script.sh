SAVEPATH=/data/wikidata_3_class_full/planRGCNpred_co
echo "start" $SECONDS >> "$SAVEPATH"/train_time.log
python3 /PlanRGCN/scripts/train/ray_tune.py \
    wikidata_3_class_full \
    $SAVEPATH\
    None
echo "done" $SECONDS >> "$SAVEPATH"/train_time.log

SAVEPATH=/data/wikidata_3_class_full/planRGCN_no_pred_co
echo "start" $SECONDS >> "$SAVEPATH"/train_time.log
python3 /PlanRGCN/scripts/train/ray_tune.py \
    wikidata_3_class_full \
    $SAVEPATH\
    None
echo "done" $SECONDS >> "$SAVEPATH"/train_time.log