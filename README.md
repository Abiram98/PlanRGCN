

# PlanRGCN

## Prerequisites
Docker should be installed

## Setup
Run the following command in the super directory where this is cloned
```
DATA_PATH=$(pwd)/PlanRGCN/data
docker run --name all_final -it -v $(pwd)/PlanRGCN:/PlanRGCN -v $(pwd)/PlanRGCN/qpp:/qpp -v $DATA_PATH:/data --shm-size=12gb ubuntu:22.04
```
when in the container, run:
```
(cd PlanRGCN && bash scripts/setup.sh)
(cd qpp && bash scripts/ini.sh)
```
The query logs with train, val, test is provided in wikidata_3_full and DBpedia_3_interval_full in data folder.
Otherwise:
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

## Extracting query plans
```
DATASET_PATH=/data/dataset_debug
bash scripts/qp/qp_extract_lsq.sh $DATASET_PATH
```

## Model Training 
First create dataset loader ...
```
...
```
The peform hyperparameter search
```
DATASET=dataset_debug
SAVEPATH ="$DATASET_PATH"/planrgcn_binner_litplan
python3 ray_tune.py $DATASET_PATH $SAVEPATH $KGSTATFOLDER
```
After hyper-parameter search, use tensorboard to verify that there isn't a model with similar validation F1 score but with fewer network parameters.

Then execute the following to get the predictions and inference time:
```
DATASET_PATH=/data/DBpedia_3_class_full
EXP_PATH=/data/DBpedia_3_class_full/planRGCNWOpredCo
python3 -m trainer.predict2 \
    -p "$EXP_PATH"/prepper.pcl \
    -m "$EXP_PATH"/best_model.pt \
    -n 3 \
    -o "$EXP_PATH"
```

To get confusion matrix results, run
```
DATASET_PATH=/data/DBpedia_3_class_full
EXP_PATH=/data/DBpedia_3_class_full/planRGCNWOpredCo

python3 /PlanRGCN/scripts/post_predict.py \
    -s $DATASET_PATH \
    -t 3 \
    -f "$EXP_PATH"/test_pred.csv \
    -a PlanRGCNWOpredCo \
    -o "$EXP_PATH"/results
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

## Processing results
```
SPLIT_DIR=/data/wikidata_3_class_full
TIMECLS=3
EXP_NAME=plan_l18192_l24096_no_pred_co
PRED_FILE="$SPLIT_DIR"/"$EXP_NAME"/test_pred.csv
APPROACH="PlanRGCN"
OUTPUTFOLDER="$SPLIT_DIR"/"$EXP_NAME"/results
python3 /PlanRGCN/scripts/post_predict.py -s $SPLIT_DIR -t $TIMECLS -f $PRED_FILE -a $APPROACH -o $OUTPUTFOLDER

```

## Starting Virtuoso Instances
The data should be loaded in.

For DBpedia,
```
CONTAINER_NAME='dbpedia_virt'
VIRT_CONFIG=/srv/data/abiram/SPARQLBench/virtuoso_setup/virtuoso_dbpedia_load_balance.ini
dbpath=/srv/data/abiram/dbpediaKG/virtuoso-db-new2/virtuoso-db-new
```


For Wikidata,
```
CONTAINER_NAME=
VIRT_CONFIG=
dbpath=/
```

For actually running the RDF store:
```
db_start (){
    CPUS="10"
    docker run --rm -v $dbpath:/database ubuntu bash -c "rm /database/virtuoso.trx"
    docker run -m 64g --rm --name $CONTAINER_NAME -d --tty --env DBA_PASSWORD=dba --env DAV_PASSWORD=dba --publish 1112:1111 --publish 8891:8890 -v $dbpath:/database -v $VIRT_CONFIG:/database/virtuoso.ini --cpus=$CPUS openlink/virtuoso-opensource-7:7.2.12
}
db_start
```


## Live predictions through terminal

```
python3 /PlanRGCN/online_predictor.py \
     --prep_path /data/DBpedia_3_class_full/plan16_10_2024/prepper.pcl\
     --model_path /data/DBpedia_3_class_full/plan16_10_2024/best_model.pt\
     --config_path /data/DBpedia_3_class_full/plan16_10_2024/model_config.json\
     --gpu yes
python3 /PlanRGCN/online_predictor.py \
     --prep_path /data/DBpedia_3_class_full/plan16_10_2024_4096_2048/prepper.pcl\
     --model_path /data/DBpedia_3_class_full/plan16_10_2024_4096_2048/best_model.pt\
     --config_path /data/DBpedia_3_class_full/plan16_10_2024_4096_2048/model_config.json\
     --gpu yes
```