#python3 /PlanRGCN/scripts/post_predict.py -s /data/wikidata_3_class_full -t 5 -f /data/wikidata_3_class_full/plan_5_int/test_pred.csv -a Plan5int -o results/ --objective objective.py

python3 /PlanRGCN/scripts/baseline_snap.py -s /data/DBpedia_3_class_full -t 5 -f /data/DBpedia_3_class_full/nn/k25/nn_test_pred.csv -o nn_results/ --objective objective.py
python3 /PlanRGCN/scripts/post_predict.py -s /data/DBpedia_3_class_full -t 5 -f nn_results/nn_pred.csv -o nn_results/ --objective objective.py

python3 /PlanRGCN/scripts/baseline_snap.py -s /data/DBpedia_3_class_full -t 5 -f /data/DBpedia_3_class_full/svm/test_pred_svm.csv -o svm_results/ --objective objective.py
python3 /PlanRGCN/scripts/post_predict.py -s /data/DBpedia_3_class_full -t 5 -f svm_results/svm_pred.csv -o svm_results/ --objective objective.py
