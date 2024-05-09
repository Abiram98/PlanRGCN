# PlanRGCN

## Prerequisites
Docker should be installed

## Setup
Run the following command in the super directory where this is cloned
```
DATA_PATH=/home/ubuntu/vol2/data_qpp2
docker run --name all_final -it -v $(pwd)/PlanRGCN:/PlanRGCN -v $(pwd)/PlanRGCN/qpp:/qpp -v $DATA_PATH:/data --shm-size=12gb ubuntu:22.04
```
when in the container, run:

(cd PlanRGCN && bash scripts/setup.sh)
(cd qpp && bash scripts/ini.sh)

```
Then untar the query log split and files:
```
tar -xvf /PlanRGCNdata/qpp_datasets.tar.gz
```
The collected MetaKG statistics are not included due to the file size limit on Github.
Nevetherless, the Virtuoso Endpoint creation is specified in /Datasets/KG

## Feature Extraction
Prerequisite: have the rdf store storing KG available.
### KG Stats
The KG stats are collected as:
```
URL=http://ENDPOINT_URL:8892/sparql
KGSTATFOLDER=/PlanRGCN/data/dbpedia2016 # where to store the extracted stat
KGSTATFOLDER=/data/planrgcn_feat/extracted_features_dbpedia2016 # where to store the extracted stat

mkdir -p "$KGSTATFOLDER"/predicate/batches
mkdir -p "$KGSTATFOLDER"/entity
#predicate features
python3 -m feature_extraction.predicates.pred_util extract-predicates -e $URL --output_dir $KGSTATFOLDER
python3 -m feature_extraction.predicates.pred_util extract-co-predicates -e $URL --input_dir $KGSTATFOLDER --output_dir "$KGSTATFOLDER"/predicate --batch_start 1 --batch_end -1
python3 -m feature_extraction.predicates.pred_stat_feat extract-predicates-stat -e $URL --input_dir $KGSTATFOLDER --output_dir "$KGSTATFOLDER" --batch_start 1 --time_log pred_freq_time_2.log --batch_end -1
python3 -m feature_extraction.predicates.pred_stat_feat extract-predicates-stat-sub-obj -e $URL --input_dir $KGSTATFOLDER --output_dir "$KGSTATFOLDER" --batch_start 1 --time_log pred_stat_subj_obj_time_2.log --batch_end -1

#entity Features
#Missign one here
python3 -m feature_extraction.entity.entity_util extract-entity-stat -e $URL --input_dir "$KGSTATFOLDER"/entity --output_dir "$KGSTATFOLDER"/entity --ent_file "$KGSTATFOLDER"/entity/entities.json --batch_start 1 --time_log ent_stat_time.log --batch_end -1

#literal Features
python3 -m feature_extraction.literal_utils distinct-literals -e $URL --output_dir "$KGSTATFOLDER"/literals --lits_file "$KGSTATFOLDER"/literals/literals.json
python3 -m feature_extraction.literal_utils extract-lits-stat -e $URL --output_dir "$KGSTATFOLDER"/literals --lits_file "$KGSTATFOLDER"/literals/literals.json --time_log lit_stat_time_1.log --batch_start 1 --batch_end -1 --timeout 1200
python3 -m feature_extraction.literal_utils extract-lits-statv2 -e $URL --output_dir "$KGSTATFOLDER"/literals --lits_file "$KGSTATFOLDER"/literals/literals.json --time_log lit_stat_time_30.log --batch_start 1 --batch_end -1 --timeout 1200
```
### Predicate Community Detection
Specify:
1. path to save predicate community features, e.g.,  "/PlanRGCN/extracted_features_wd/predicate/pred_co"
2. path to extracted predicate cooccurence features, e.g., "/PlanRGCN/extracted_features_wd/predicate/predicate_cooccurence/batch_response/"
```
python3 /PlanRGCN/scripts/com_feat.py 
```


## Extracting query plans
```
DATASET_PATH=/data/wikidata_0_1_10_weightloss
bash scripts/qp/qp_extract_lsq.sh $DATASET_PATH
```

## Model Training 
```
DATASET=dataset_debug
SAVEPATH ="$DATASET_PATH"/planrgcn_binner_litplan
python3 ray_tune.py $DATASET_PATH $SAVEPATH $KGSTATFOLDER
```

Prediction quality is analyzed in the notebooks in the notebook folder.

## Load Balancing
Start the RDF store with the Virtuoso RDF store init files in virt_feat_conf folder.
To run the experiment execute the following:
```
(cd $LB/load_balance_SVM_44_10_workers && timeout -s 2 7200 python3 -m load_balance.main_balancer config.conf)
```

## Baseline Methods
The Baseline method's are explained in the qpp folder's README.



# Updated training scheme
First create the datasets:
```
mkdir -p /data/wikidata_3_class_full/planRGCNpred_co
python3 /PlanRGCN/scripts/train/dataset_creator.py wikidata_3_class_full /data/wikidata_3_class_full/planRGCNpred_co /data/metaKGStat/wikidata yes
```
then train a model:
python3 /PlanRGCN/scripts/train/ray_tune.py wikidata_3_class_full /data/wikidata_3_class_full/planRGCNpred_co