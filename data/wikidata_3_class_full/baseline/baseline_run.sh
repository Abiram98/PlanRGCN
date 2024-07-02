DATASET=/data/wikidata_3_class_full
{ time python3 -m qpp_new.trainer svm  --data-dir $DATASET --results-dir $DATASET > $DATASET/svm.log ; } 2> svm_train.log
echo "SVM"  $SECONDS
{ time python3 -m qpp_new.trainer nn  --data-dir $DATASET --results-dir $DATASET > $DATASET/nn.log ; } 2> nn_train.log
echo "NN"  $SECONDS
