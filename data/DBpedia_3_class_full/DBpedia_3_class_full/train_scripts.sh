echo DBpedia train
FEAT=/data/metaKGStat/dbpedia
EXP=/data/DBpedia_3_class_full/plan_5_int
mkdir -p $EXP
echo """
import numpy as np

def cls_func(lat):
    vec = np.zeros(5)
    if lat < 0.004:
        vec[0] = 1
    elif (0.004 < lat) and (lat <= 1):
        vec[1] = 1
    elif (1 < lat) and (lat <= 10):
        vec[2] = 1
    elif (10 < lat) and (lat <= 899):
        vec[3] = 1
    elif 899 < lat:
        vec[4] = 1
    return vec

n_classes = 5
name_dict ={
                0: '(0s; 0.004]',
                1: '(0.004s; 1]',
                2: '(1s; 10]',
                3: '(10; timeout=15min]',
                4: 'timeout',
            } 

"""> "$EXP"/objective.py
python3 /PlanRGCN/scripts/train/ray_run.py DBpedia_3_class_full $EXP --feat_path $FEAT --class_path "$EXP"/objective.py
python3 -m trainer.predict2 -p "$EXP"/prepper.pcl -m "$EXP"/best_model.pt -n 5 -o "$EXP" --l1 4096 --l2 4096

EXP=/data/DBpedia_3_class_full/plan_5_int_5min
mkdir -p $EXP
echo """
import numpy as np
def cls_func(lat):
    vec = np.zeros(5)
    if lat < 0.004:
        vec[0] = 1
    elif (0.004 < lat) and (lat <= 1):
        vec[1] = 1
    elif (1 < lat) and (lat <= 10):
        vec[2] = 1
    elif (10 < lat) and (lat <= 300):
        vec[3] = 1
    elif 300 < lat:
        vec[4] = 1
    return vec

n_classes = 5
name_dict ={
                0: '(0s; 0.004]',
                1: '(0.004s; 1]',
                2: '(1s; 10]',
                3: '(10; timeout=5min]',
                4: 'timeout',
            } 


"""> "$EXP"/objective.py
python3 /PlanRGCN/scripts/train/ray_run.py DBpedia_3_class_full $EXP --feat_path $FEAT --class_path "$EXP"/objective.py
python3 -m trainer.predict2 -p "$EXP"/prepper.pcl -m "$EXP"/best_model.pt -n 5 -o "$EXP" --l1 4096 --l2 4096
