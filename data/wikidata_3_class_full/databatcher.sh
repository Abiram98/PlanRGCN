mkdir -p /data/wikidata_3_class_full/planRGCNpred_co
python3 /PlanRGCN/scripts/train/dataset_creator.py wikidata_3_class_full /data/wikidata_3_class_full/planRGCNpred_co /data/metaKGStat/wikidata yes
echo $SECONDS
mkdir -p /data/wikidata_3_class_full/planRGCN_no_pred_co
python3 /PlanRGCN/scripts/train/dataset_creator.py wikidata_3_class_full /data/wikidata_3_class_full/planRGCN_no_pred_co /data/metaKGStat/wikidata no
echo $SECONDS "S"