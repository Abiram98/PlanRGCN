NN_PRED_FILE=/data/DBpedia_3_class_full/nn/k25/nn_test_pred.csv
SVM_PRED_FILE=/data/DBpedia_3_class_full/svm/test_pred_svm.csv
PLAN_EXP_PATH=/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co
PLAN_PRED=$PLAN_EXP_PATH/test_pred.csv
EXP_NAME=plan_l14096_l21025_no_pred_co
EXP_PATH=/data/DBpedia_3_class_full
N_CLASSES=3


python3 /PlanRGCN/scripts/baseline_snap.py -s $EXP_PATH -t $N_CLASSES -f $SVM_PRED_FILE -o $PLAN_EXP_PATH/svm_results/ --objective $PLAN_EXP_PATH/objective.py
python3 /PlanRGCN/scripts/post_predict_unseen.py -s $EXP_PATH -t $N_CLASSES -f $PLAN_EXP_PATH/svm_results/svm_pred.csv -a SVM -o $PLAN_EXP_PATH/svm_results/ --objective $PLAN_EXP_PATH/objective.py
python3 /PlanRGCN/scripts/post_predict_unseenPartially.py -s $EXP_PATH -t $N_CLASSES -f $PLAN_EXP_PATH/svm_results/svm_pred.csv -a SVM -o $PLAN_EXP_PATH/svm_results/ --objective $PLAN_EXP_PATH/objective.py
