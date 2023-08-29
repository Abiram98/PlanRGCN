'''module contains methods pertaining testing the speedup and other workload related matters'''
import numpy as np

def avg_runtime(data):
    count, run_times = 0,[]
    oracle = []
    for k in data.keys():
        if not 'prediction' in data[k].keys():
            continue
        count += 1
        
        #const_time =  data[k]['bgp_construction_duration']+ data[k]['tps_const_duration']+ data[k]['inference_time']
        pred = True if data[k]['prediction'] == 1 else False
        if pred:
            run_times.append(data[k]['bloom_runtime'])
        else:
            run_times.append(data[k]['jena_runtime'])
        if data[k]['bloom_runtime'] > data[k]['jena_runtime']:
            oracle.append(data[k]['jena_runtime'])
        else:
            oracle.append(data[k]['bloom_runtime'])
    return run_times,oracle, np.sum(run_times)/count,np.sum(oracle)/count

def get_runtime(data):
    count, run_times = 0,[]
    oracle = []
    for k in data.keys():
        if not 'prediction' in data[k].keys():
            continue
        count += 1
        
        #const_time =  data[k]['bgp_construction_duration']+ data[k]['tps_const_duration']+ data[k]['inference_time']
        pred = True if data[k]['prediction'] == 1 else False
        if pred:
            run_times.append(data[k]['bloom_runtime'])
        else:
            run_times.append(data[k]['jena_runtime'])
        if data[k]['bloom_runtime'] > data[k]['jena_runtime']:
            oracle.append(data[k]['jena_runtime'])
        else:
            oracle.append(data[k]['bloom_runtime'])
    return run_times, oracle

def get_runtimes_wo_pred(data, leaf=False):
    jena_run_times, blf = [], []
    for k in data.keys():
        if not 'prediction' in data[k].keys():
            continue
        jena_run_times.append(data[k]['jena_runtime'])
        blf.append(data[k]['bloom_runtime'])
    return jena_run_times, blf


def avg_runtime_type(data, leaf=False):
    count, jena_run_times, blf = 0,[], []
    lf = []
    for k in data.keys():
        if not 'prediction' in data[k].keys():
            continue
        count += 1
        jena_run_times.append(data[k]['jena_runtime'])
        blf.append(data[k]['bloom_runtime'])
        if leaf:
            lf.append(data[k]['leapfrog'])
    if leaf:    
        return jena_run_times, blf, np.sum(jena_run_times)/count, np.sum(blf)/count, lf
    else:
        return jena_run_times, blf, np.sum(jena_run_times)/count, np.sum(blf)/count

def cum_runtime(data):
    count, run_times = 0,[]
    for k in data.keys():
        count += 1
        if not 'prediction' in data[k].keys():
            continue
        pred = True if data[k]['prediction'] == 1 else False
        if pred:
            run_times.append(data[k]['bloom_runtime'])
        else:
            run_times.append(data[k]['jena_runtime'] )
            #run_times.append(data[k]['jena_runtime'])
    return np.sum(run_times)

def cum_runtime_type(data, leaf=False):
    count, jena_run_times, blf = 0,[], []
    lf = []
    oracle = []
    for k in data.keys():
        if not 'prediction' in data[k].keys():
            continue
        count += 1
        jena_run_times.append(data[k]['jena_runtime'])
        blf.append(data[k]['bloom_runtime'])
        if leaf:
            lf.append(data[k]['leapfrog'])
        #const_time = data[k]['bgp_construction_duration']+ data[k]['tps_const_duration']+ data[k]['inference_time']
        if data[k]['bloom_runtime'] > data[k]['jena_runtime']:
            oracle.append(data[k]['jena_runtime'])
        else:
            oracle.append(data[k]['bloom_runtime'])
    if leaf:
        return np.sum(jena_run_times), np.sum(blf), np.sum(lf), np.sum(oracle)
    else:
        return np.sum(jena_run_times), np.sum(blf), np.sum(oracle)

def get_BF_run_times(data):
    runtimes = []
    oracle = []
    jena = []
    for x in data.keys():
        if not 'prediction' in data[x].keys():
            continue
        if data[x]['gt']:
            if data[x]['bloom_runtime'] > data[x]['jena_runtime']:
                oracle.append(data[x]['jena_runtime'])
            else:
                oracle.append(data[x]['bloom_runtime'])
            runtimes.append(data[x]['bloom_runtime'])
            jena.append(data[x]['jena_runtime'])
    return runtimes, oracle, jena