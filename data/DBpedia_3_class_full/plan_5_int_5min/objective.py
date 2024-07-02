
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

