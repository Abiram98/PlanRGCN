# assumes that the distances has already been calculated.
DISTANCES="/data/wikidata_0_1_10_v2_weight_loss_retrain/wikidata_0_1_10_v2_weight_loss/distances/distances/"
OUTDIR="/data/wikidata_0_1_10_v2_weight_loss_retrain/wikidata_0_1_10_v2_weight_loss"
DBPATH="/data/wikidata_0_1_10_v2_weight_loss_retrain/wikidata_0_1_10_v2_weight_loss/ged.db"


#algebra features:
{ time python3 -m qpp_features.feature_generator /data/wikidata_0_1_10_v2_weight_loss_retrain/wikidata_0_1_10_v2_weight_loss/all.tsv $OUTDIR -t alg_feat 1> "$OUTDIR"/baseline/alg_feat_time.log ; } 2> "$OUTDIR"/baseline/alg_feat_time.errlog
rm *txt

#GED feature calculation
: '
{ time python3 -c """
from qpp_new.feature_combiner import create_different_k_ged_dist_matrix
create_different_k_ged_dist_matrix(basedir='$OUTDIR', database_path='$DBPATH')
"""; } 2> test.log
'

