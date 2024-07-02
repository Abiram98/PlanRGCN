
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


