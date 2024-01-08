import os
import load_balance.arrival_time
from load_balance.arrival_time import ArrivalRateDecider
import pandas as pd
from  load_balance.workload import Workload
import copy
import multiprocessing
import time, datetime
from SPARQLWrapper import SPARQLWrapper, JSON
import random
import numpy as np
import json

def get_all_queries(b):
    queries = []
    for w in b.slow_queue.worker:
        queries.extend(w.past_queries)
    for w in b.med_queue.worker:
        queries.extend(w.past_queries)
    for w in b.fast_queue.worker:
        queries.extend(w.past_queries)
    return queries

def get_all_queriesFIFO(b):
    queries = []
    for w in b.fast_queue.worker:
        queries.extend(w.past_queries)
    return queries

def workload_latency(qs):
    total_latency = 0
    for q in qs:
        total_latency += (q.finish_time- q.arrivaltime)
    return total_latency
def relative_improvement(wl, bwl):
    return (bwl-wl)/bwl

def execute_query(query, url):
    sparql = SPARQLWrapper(url)
    sparql.setReturnFormat(JSON)
    sparql.setTimeout(1800)
    sparql.setQuery(query)
    try:
        ret = sparql.query()        
    except Exception as e:
        return None
    return ret
    
def execute_query_worker(workload:Workload, w_type, url, start_time, path):
    w_str = w_type
    w_type = "med" if w_type.startswith('med') else w_type
    w_type = "fast" if w_type.startswith('fast') else w_type
    w_type = "slow" if w_type.startswith('slow') else w_type
    data = []
    match w_type:
        case "slow":
            while True:
                val = workload.slow_queue.get()
                if val is None:
                    break
                (q,ar) = val
                n_ar = ar+start_time
                while( time.time()< n_ar):
                    pass
                q_start_time = time.time()
                #execute stuff
                ret = execute_query(q.query_string,url)
                q_end_time = time.time()
                elapsed_time = q_end_time-q_start_time
                data.append({'query': str(q), 'start_time':start_time, 'arrival_time': n_ar, 'query_start_time': q_start_time, 'query_end_time': q_end_time, 'elapsed_time': elapsed_time, 'response': 'ok' if ret is not None else 'not ok'})
            with open(f"{path}/{w_str}.json", 'w') as f:
                json.dump(data,f)
            exit()
        case "med":
            while True:
                val = workload.med_queue.get()
                if val is None:
                    break
                (q,ar) = val
                n_ar = ar+start_time
                while( time.time()< n_ar):
                    pass
                q_start_time = time.time()
                #execute stuff
                ret = execute_query(q.query_string,url)
                q_end_time = time.time()
                elapsed_time = q_end_time-q_start_time
                data.append({'query': str(q), 'start_time':start_time, 'arrival_time': n_ar, 'query_start_time': q_start_time, 'query_end_time': q_end_time, 'elapsed_time': elapsed_time, 'response': 'ok' if ret is not None else 'not ok'})
            with open(f"{path}/{w_str}.json", 'w') as f:
                json.dump(data,f)
                
            exit()
        case "fast":
            while True:
                val = workload.fast_queue.get()
                if val is None:
                    break
                (q,ar) = val
                n_ar = ar+start_time
                while( time.time()< n_ar):
                    pass
                q_start_time = time.time()
                #execute stuff
                ret = execute_query(q.query_string,url)
                q_end_time = time.time()
                elapsed_time = q_end_time-q_start_time
                data.append({'query': str(q), 'start_time':start_time, 'arrival_time': n_ar, 'query_start_time': q_start_time, 'query_end_time': q_end_time, 'elapsed_time': elapsed_time, 'response': 'ok' if ret is not None else 'not ok'})
            with open(f"{path}/{w_str}.json", 'w') as f:
                json.dump(data,f)
            exit()
            

def main_balance_runner(sample_name, scale, url = 'http://172.21.233.23:8891/sparql'):
    np.random.seed(42)
    random.seed(42)
    
    sample_name="wikidata_0_1_10_v2_path_weight_loss"
    scale="planrgcn_binner"
    url = "http://172.21.233.14:8891/sparql"
    
    # Workload Setup
    df = pd.read_csv(f'/data/{sample_name}/test_sampled.tsv', sep='\t')
    print(df)
    print(df['mean_latency'].quantile(q=0.25))
    w = Workload()
    w.load_queries(f'/data/{sample_name}/test_sampled.tsv')
    w.set_time_cls(f"/data/{sample_name}/{scale}/test_pred.csv")
    a = ArrivalRateDecider()
    w.shuffle_queries()
    w.shuffle_queries()
    w.reorder_queries()
    w.set_arrival_times(a.assign_arrival_rate(w, mu=1000))
    w.initialise_queues()
    
    w.slow_queue.put(None)
    w.med_queue.put(None)
    w.med_queue.put(None)
    w.med_queue.put(None)
    w.fast_queue.put(None)
    w.fast_queue.put(None)
    w.fast_queue.put(None)
    w.fast_queue.put(None)
    start_time = time.time()
    print(start_time)
    f_lb =f'/data/{sample_name}/load_balance'
    os.system(f'mkdir -p {f_lb}')
    path = f_lb
    procs = {
        "slow" : multiprocessing.Process(target=execute_query_worker, args=(w,"slow",url, start_time,path,)),
        "med1" : multiprocessing.Process(target=execute_query_worker, args=(w,"med1",url, start_time,path,)),
        "med2" : multiprocessing.Process(target=execute_query_worker, args=(w,"med2",url, start_time,path,)),
        "med3" : multiprocessing.Process(target=execute_query_worker, args=(w,"med3",url, start_time,path,)),
        "fast1" : multiprocessing.Process(target=execute_query_worker, args=(w,"fast1",url, start_time,path,)),
        "fast2" : multiprocessing.Process(target=execute_query_worker, args=(w,"fast2",url, start_time,path,)),
        "fast3" : multiprocessing.Process(target=execute_query_worker, args=(w,"fast3",url, start_time,path,)),
        "fast4" : multiprocessing.Process(target=execute_query_worker, args=(w,"fast4",url, start_time,path,)),
    }
    
    for k in procs.keys():
        procs[k].start()
    
    for k in procs.keys():
        procs[k].join()
    end_time = time.time()
    print(f"elapsed time: {end_time-start_time}")

def execute_query_worker_FIFO(workload:Workload, w_type, url, start_time, path):
    w_str = w_type
    data = []
    #debug varible
    #qs = []
    while True:
        val = workload.FIFO_queue.get()
        if val is None:
            break
        (q,ar) = val
        n_ar = ar+start_time
        while( time.time()< n_ar):
            pass
        q_start_time = time.time()
        #execute stuff
        ret = execute_query(q.query_string,url)
        q_end_time = time.time()
        elapsed_time = q_end_time-q_start_time
        print(w_type+ '  ', len(data))
        #qs.append(q)
        data.append({'query': str(q), 'start_time':start_time, 'arrival_time': n_ar, 'query_start_time': q_start_time, 'query_end_time': q_end_time, 'elapsed_time': elapsed_time, 'response': 'ok' if ret is not None else 'not ok'})
    with open(f"{path}/{w_str}.json", 'w') as f:
        json.dump(data,f)
    #debug variable
    """chosen_i = None
    for i, d in enumerate(qs):
        if d.true_time_cls == 2:
            chosen_i = i
            break
    print(w_type,chosen_i)"""
    exit()

def main_balance_runnerFIFO(sample_name, scale, url = 'http://172.21.233.23:8891/sparql'):
    np.random.seed(42)
    random.seed(42)
    
    sample_name="wikidata_0_1_10_v2_path_weight_loss"
    scale="planrgcn_binner"
    url = "http://172.21.233.14:8891/sparql"
    
    # Workload Setup
    df = pd.read_csv(f'/data/{sample_name}/test_sampled.tsv', sep='\t')
    print(df)
    print(df['mean_latency'].quantile(q=0.25))
    w = Workload()
    w.load_queries(f'/data/{sample_name}/test_sampled.tsv')
    w.set_time_cls(f"/data/{sample_name}/{scale}/test_pred.csv")
    a = ArrivalRateDecider()
    w.shuffle_queries()
    w.shuffle_queries()
    w.reorder_queries()
    w.set_arrival_times(a.assign_arrival_rate(w, mu=1000))
    w.initialise_FIFO_que()
    for _ in range(8):
        w.FIFO_queue.put(None)
    start_time = time.time()
    print(start_time)
    f_lb =f'/data/{sample_name}/load_balance_FIFO'
    os.system(f'mkdir -p {f_lb}')
    path = f_lb
    procs = [
        multiprocessing.Process(target=execute_query_worker_FIFO, args=(w,"slow",url, start_time,path,)),
        multiprocessing.Process(target=execute_query_worker_FIFO, args=(w,"med1",url, start_time,path,)),
        multiprocessing.Process(target=execute_query_worker_FIFO, args=(w,"med2",url, start_time,path,)),
        multiprocessing.Process(target=execute_query_worker_FIFO, args=(w,"med3",url, start_time,path,)),
        multiprocessing.Process(target=execute_query_worker_FIFO, args=(w,"fast1",url, start_time,path,)),
        multiprocessing.Process(target=execute_query_worker_FIFO, args=(w,"fast2",url, start_time,path,)),
        multiprocessing.Process(target=execute_query_worker_FIFO, args=(w,"fast3",url, start_time,path,)),
        multiprocessing.Process(target=execute_query_worker_FIFO, args=(w,"fast4",url, start_time,path,)),
    ]
    
    for k in procs:
        k.start()
    
    for k in procs:
        k.join()
    end_time = time.time()
    print(f"elapsed time: {end_time-start_time}")


if __name__ == "__main__":
    sample_name="wikidata_0_1_10_v2_path_weight_loss"
    scale="planrgcn_binner"
    #main_balance_runner(sample_name, scale)
    main_balance_runnerFIFO(sample_name, scale)
    exit()
    
    df = pd.read_csv(f'/data/{sample_name}/test_sampled.tsv', sep='\t')
    print(df.head(5))
    print(df['mean_latency'].quantile(q=0.25))
    w = Workload()
    w.load_queries(f'/data/{sample_name}/test_sampled.tsv')
    w.set_time_cls(f"/data/{sample_name}/{scale}/test_pred.csv")
    a = ArrivalRateDecider()
    w.set_arrival_times(a.assign_arrival_rate(w, mu=10000000))
    w.shuffle_queries()
    w.shuffle_queries()
    w.reorder_queries()
    exit()
    
    w2 = copy.deepcopy(w)
    print(f"Queries finished before concurrency: {w.queries_finished_before_other()}")
    print(f"Queries finished before concurrency: {w2.queries_finished_before_other()}")
    from load_balance.balancer import Balancer,FIFOBalancer
    print("\nClassifier Simulation")
    
    b = Balancer(w)     
    b.run()
    qs = get_all_queries(b)
    w_lat = workload_latency(qs)
    print(f"Average workload latency {w_lat/len(qs)}")
    print(f"Workload latency {w_lat}")
    print("\nFIFO Simulation")
    b = FIFOBalancer(w2)     
    b.run()
    qs = get_all_queriesFIFO(b)
    w_lat_fifo = workload_latency(qs)
    print(f"Average workload latency {w_lat_fifo/len(qs)}")
    print(f"Workload latency {w_lat_fifo}")
    print(f"relative difference {relative_improvement(w_lat, w_lat_fifo)}")
    
