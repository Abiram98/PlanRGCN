url=http://130.225.39.154:8890/sparql
task=extract-predicates
output_dir=/PlanRGCN/extracted_features
pred_file=predicates.json

#python3 -m feature_extraction.predicates.pred_util $task -e $url --output_dir $output_dir --pred_file $pred_file
task=extract-co-predicates
python3 -m feature_extraction.predicates.pred_util $task -e $url --output_dir $output_dir --pred_file $pred_file