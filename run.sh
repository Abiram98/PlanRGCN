#task=extract-predicates-query-log
#queries=/SPARQLBench/dbpedia2015_16/ordered_queries2015_2016_clean_w_stat.tsv
#outputPath=/PlanRGCN/extracted_features/predicate/predicate_in_dbpedia2016.json
#task=test
#(mvn exec:java -f "/PlanRGCN/qpe/pom.xml" -Dexec.args="$task $queries $outputPath")
queries=/qpp/dataset/queries2015_2016_clean_w_stat_q_str.tsv
outputPath=/PlanRGCN/extracted_features/queryplans
task=extract-query-plans
(mvn exec:java -f "/PlanRGCN/qpe/pom.xml" -Dexec.args="$task $queries $outputPath")
#bash run.sh > query_plan_gen_log.txt 2>&1
