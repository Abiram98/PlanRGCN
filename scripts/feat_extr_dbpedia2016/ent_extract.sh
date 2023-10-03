url=http://130.225.39.154:8890/sparql
task=extract-entity-stat
output_dir=/PlanRGCN/extracted_features/entities
pred_file=/PlanRGCN/extracted_features/entities/entities_in_dbpedia2016.json
input_dir=/PlanRGCN/extracted_features/entities
batch_start=1
batch_end=244

python3 -m feature_extraction.entity.entity_util $task -e $url --output_dir $output_dir --ent_file $pred_file --batch_start $batch_start --batch_end $batch_end

url=http://172.21.233.23:8890/sparql/