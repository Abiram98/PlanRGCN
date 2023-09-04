queries=/SPARQLBench/dbpedia2015_16/ordered_queries2015_2016_clean_w_stat.tsv
task=extract-predicates-query-log
outputPath=/PlanRGCN/extracted_features/predicate/predicate_in_dbpedia2016.json
(mvn exec:java -f "/PlanRGCN/qpe/pom.xml" -Dexec.args="$task $queries $outputPath")
