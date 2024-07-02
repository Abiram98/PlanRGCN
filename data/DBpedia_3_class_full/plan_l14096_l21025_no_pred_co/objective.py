import numpy as np

def cls_func(lat):
    vec = np.zeros(3)
    if lat < 1:
        vec[0] = 1
    elif (1 < lat) and (lat < 10):
        vec[1] = 1
    elif 10 < lat:
        vec[2] = 1
    return vec

name_dict = {
                0: '(0; 1]',
                1: '(1; 10]',
                2: '(10; $\\infty$',
            }

n_classes = 3
