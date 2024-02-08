# Feature Extraction

## Prerequisite:
A SPARQL endpoint with the KG needs to be accessible.

## Predicate Feature Extraction
### Predicates
```
WIKIDATA=/PlanRGCN/data/wikidata
python3 -m feature_extraction.predicates.pred_util extract-predicates -e http://172.21.233.23:8891/sparql --output_dir $WIKIDATA
```
### Predicate Co-Occurence Features
```
mkdir "$WIKIDATA"/predicate/batches
python3 -m feature_extraction.predicates.pred_util extract-co-predicates \
-e http://172.21.233.23:8891/sparql \
--input_dir $WIKIDATA \
--output_dir "$WIKIDATA"/predicate \
--batch_start 1 --batch_end -1
```

