wikidata_qpp2 (){
    #config_path=/data/abiram/SPARQLBench/virtuoso_setup/virtuoso_WD_load_balance.ini
    CPUS="10"
    dbpath=/data/abiram/wdbench/virtuoso_dabase
    imp_path=/data/abiram/wdbench/import
    docker run --rm -v $dbpath:/database ubuntu bash -c "rm /database/virtuoso.trx"
    #-v $config_path:/database/virtuoso.ini
    docker run -m 64g --rm --name wdbench_virt -d --tty --env DBA_PASSWORD=dba --env DAV_PASSWORD=dba --publish 1112:1111 --publish 8891:8890 -v $dbpath:/database -v imp_path:/import --cpus=$CPUS openlink/virtuoso-opensource-7:7.2.12
}

STOPTIME=13
STARTIME=5

wikidata_qpp2
sleep $STARTIME
#docker stop wdbench_virt
#sleep $STOPTIME
#wikidata_qpp2
#sleep $STARTIME
START=$SECONDS
DATAPATH='/data/abiram/data_qpp'
PREDICTION_FILE='/data/DBpedia_3_class_full/nn/test_pred.csv'
SPLIT_FILE='/data/DBpedia_3_class_full/test_sampled.tsv'
PRED_COL='nn_prediction'
OUTPUT_FOLDER='/data/DBpedia_3_class_full/admission_control/nn_44'
ARRIVAL_RATE=44
ENDPOINT=http://172.21.233.14:8891/sparql
WORKERS=10

docker run --rm --name adm_ctrl -v /data/abiram/data_qpp:/data plan:2 \
  timeout -s 2 7200 python3 -m load_balance.admission_controller \
  -f $SPLIT_FILE \
  -p $PREDICTION_FILE \
  -t $PRED_COL \
  -o $OUTPUT_FOLDER -r $ARRIVAL_RATE \
  -u $ENDPOINT \
  -i $WORKERS \
  --seed 42 \
  --interval 2 \
  -l no

echo "Started at $START - NN finished after $SECONDS"

docker stop wdbench_virt
sleep $STOPTIME
wikidata_qpp2
sleep $STARTIME

START=$SECONDS
DATAPATH='/data/abiram/data_qpp'
PREDICTION_FILE='/data/DBpedia_3_class_full/svm/test_pred_cls.csv'
SPLIT_FILE='/data/DBpedia_3_class_full/test_sampled.tsv'
PRED_COL='svm_prediction'
OUTPUT_FOLDER='/data/DBpedia_3_class_full/admission_control/svm_44'
ARRIVAL_RATE=44
ENDPOINT=http://172.21.233.14:8891/sparql
WORKERS=10

docker run --rm --name adm_ctrl -v /data/abiram/data_qpp:/data plan:2 \
  timeout -s 2 7200 python3 -m load_balance.admission_controller \
  -f $SPLIT_FILE \
  -p $PREDICTION_FILE \
  -t $PRED_COL \
  -o $OUTPUT_FOLDER -r $ARRIVAL_RATE \
  -u $ENDPOINT \
  -i $WORKERS \
  --seed 42 \
  --interval 2 \
  -l no

echo "Started at $START - SVM finished after $SECONDS"
docker stop wdbench_virt


exit
#already run correctly presumably, Make analysis to be sure.
DATAPATH='/data/abiram/data_qpp'
PREDICTION_FILE='/data/DBpedia_3_class_full/plan_l14096_l21025_no_pred_co/test_pred.csv'
SPLIT_FILE='/data/DBpedia_3_class_full/test_sampled.tsv'
PRED_COL='planrgcn_prediction'
OUTPUT_FOLDER='/data/DBpedia_3_class_full/admission_control/planrgcn_44'
ARRIVAL_RATE=44
ENDPOINT=http://172.21.233.14:8891/sparql
WORKERS=10

docker run --rm --name adm_ctrl -v /data/abiram/data_qpp:/data plan:2 \
  timeout -s 2 7200 python3 -m load_balance.admission_controller \
  -f $SPLIT_FILE \
  -p $PREDICTION_FILE \
  -t $PRED_COL \
  -o $OUTPUT_FOLDER -r $ARRIVAL_RATE \
  -u $ENDPOINT \
  -i $WORKERS \
  --seed 42 \
  --interval 2