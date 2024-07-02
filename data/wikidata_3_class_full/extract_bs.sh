# assumes that the distances has already been calculated.
DATASET="/data/wikidata_3_class_full"
DISTANCE="$DATASET"/"distances"

bash /PlanRGCN/qpp/scripts/baseline_feat_const.sh $DATASET $DISTANCE > feature.log
