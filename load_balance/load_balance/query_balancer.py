import os
from load_balance.workload.arrival_time import ArrivalRateDecider
import pandas as pd
from  load_balance.workload.workload import Workload, WorkloadV2, WorkloadV3
import multiprocessing
import time, datetime
from SPARQLWrapper import SPARQLWrapper, JSON
import random
import numpy as np
import json
from load_balance.query_balancer_v1 import *
from multiprocessing import Array, Value
from load_balance.workload.query import Query
import load_balance.const as const

class Worker:
    def __init__(self, workload:WorkloadV3,w_type, url, start_time, path, timeout=900):
        self.workload = workload
        self.path = path
        self.start_time = start_time
        self.w_type:str = w_type
        self.setup_sparql(url, timeout)
    
    def setup_sparql(self, url, timeout):
        self.sparql = SPARQLWrapper(url, defaultGraph='http://localhost:8890/dataspace')
        
        self.sparql.setReturnFormat(JSON)
        #sparql.setTimeout(1800)
        self.sparql.setTimeout(timeout)
    
    def execute_query(self, query):
        self.sparql.setQuery(query)
        try:
            ret = self.sparql.query()        
        except TimeoutError:
            return 1
        except Exception as e:
            return e
        return ret

    def execute_query_worker(self):
        workload = self.workload
        w_type = self.w_type
        
        w_str = self.w_type
        w_type = "med" if w_type.startswith('med') else w_type
        w_type = "fast" if w_type.startswith('fast') else w_type
        w_type = "slow" if w_type.startswith('slow') else w_type
        data = []
        #debug code
        ret = None
        try:
            match w_type:
                case "slow":
                    while True:
                        val = workload.slow_queue.get()
                        if val is None:
                            break
                        q = val
                        q:Query
                        q_start_time = time.time()
                        
                        #execute stuff
                        ret = self.execute_query(q.query_string)
                        q_end_time = time.time()
                        elapsed_time = q_end_time-q_start_time
                        data.append({
                            'query': str(q), 
                            'start_time':self.start_time, 
                            'arrival_time': q.arrival_time, 
                            'queue_arrival_time':q.queue_arrival_time, 
                            'query_execution_start': q_start_time, 
                            'query_execution_end': q_end_time, 
                            'execution_time': elapsed_time, 
                            'response': ret.message if isinstance(ret, Exception) else 'timed out' if ret == 1 else 'ok'})
                case "med":
                    while True:
                        val = workload.med_queue.get()
                        if val is None:
                            break
                        q = val
                        q:Query
                        q_start_time = time.time()
                        
                        #execute stuff
                        ret = self.execute_query(q.query_string)
                        q_end_time = time.time()
                        elapsed_time = q_end_time-q_start_time
                        data.append({
                            'query': str(q), 
                            'start_time':self.start_time, 
                            'arrival_time': q.arrival_time, 
                            'queue_arrival_time':q.queue_arrival_time, 
                            'query_execution_start': q_start_time, 
                            'query_execution_end': q_end_time, 
                            'execution_time': elapsed_time, 
                            'response': ret.message if isinstance(ret, Exception) else 'timed out' if ret == 1 else 'ok'})
                case "fast":
                    while True:
                        val = workload.fast_queue.get()
                        if val is None:
                            break
                        q = val
                        q:Query
                        q_start_time = time.time()
                        
                        #execute stuff
                        ret = self.execute_query(q.query_string)
                        q_end_time = time.time()
                        elapsed_time = q_end_time-q_start_time
                        data.append({
                            'query': str(q), 
                            'start_time':self.start_time, 
                            'arrival_time': q.arrival_time, 
                            'queue_arrival_time':q.queue_arrival_time, 
                            'query_execution_start': q_start_time, 
                            'query_execution_end': q_end_time, 
                            'execution_time': elapsed_time, 
                            'response': ret.message if isinstance(ret, Exception) else 'timed out' if ret == 1 else 'ok'})
            
            with open(f"{self.path}/{w_str}.json", 'w') as f:
                json.dump(data,f)
        except KeyboardInterrupt:
            with open(f"{self.path}/{w_str}.json", 'w') as f:
                json.dump(data,f)
        exit()

def dispatcher(workload: WorkloadV3, start_time, path):
    try:
        for numb, (q, a) in enumerate(zip(workload.queries, workload.arrival_times)):
            if numb % 100 == 0:
                s = {}
                s['fast'] = workload.fast_queue.qsize()
                s['med'] = workload.med_queue.qsize()
                s['slow'] = workload.slow_queue.qsize()
                s['time'] = time.time() - start_time
                print(f"Main process: query {numb} / {len(workload.queries)}: {s}")
            n_arr = start_time + a
            q.arrival_time = n_arr
            if n_arr > time.time():
                time.sleep(n_arr - time.time())
            
            match q.time_cls:
                case 0:
                    q.queue_arrival_time = time.time()
                    workload.fast_queue.put(q)
                    #workload.queue_dct[fast_keys[fast_idx]].put(q)
                    #fast_idx = (fast_idx+1) % len(fast_keys)
                case 1:
                    q.queue_arrival_time = time.time()
                    workload.med_queue.put(q)
                    #workload.queue_dct[medium_keys[med_idx]].put(q)
                    #med_idx = (med_idx+1) % len(medium_keys)
                case 2:
                    q.queue_arrival_time = time.time()
                    workload.slow_queue.put(q)
                    #workload.queue_dct['slow'].put(q)
        #for k in workload.queue_dct.keys():
        #    workload.queue_dct[k].put(None)
        
        workload.slow_queue.put(None)
        workload.med_queue.put(None)
        workload.med_queue.put(None)
        workload.med_queue.put(None)
        workload.fast_queue.put(None)
        workload.fast_queue.put(None)
        workload.fast_queue.put(None)
        workload.fast_queue.put(None)
        with open(f"{path}/main.json", 'w') as f:
            f.write("done")
    except KeyboardInterrupt:
        s = {}
        s['fast'] = workload.fast_queue.qsize()
        s['med'] = workload.med_queue.qsize()
        s['slow'] = workload.slow_queue.qsize()
        s['time'] = time.time() - start_time
        print(f"Main process: query {numb} / {len(workload.queries)}: {s}")
        with open(f"{path}/main.json", 'w') as f:
            f.write("done")
        exit()           
    exit()


def main_balance_runner(sample_name, scale, url = 'http://172.21.233.23:8891/sparql'):
    np.random.seed(42)
    random.seed(42)
    
    #sample_name="wikidata_0_1_10_v2_path_weight_loss"
    #scale="planrgcn_binner"
    #url = "http://172.21.233.14:8891/sparql"
    
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
    w.set_arrival_times(a.assign_arrival_rate(w, mu=const.MU))
    
    f_lb =f'/data/{sample_name}/load_balance'
    os.system(f'mkdir -p {f_lb}')
    path = f_lb
    procs = {}
    work_names = ["slow","med1", "med2","med3","fast1","fast2","fast3","fast4" ]
    start_time = time.time()
    for work_name in work_names:
        procs[work_name] = multiprocessing.Process(target=Worker(w,work_name,url, start_time,path).execute_query_worker)
    procs['main'] = multiprocessing.Process(target=dispatcher, args=(w, start_time, path,))
    
    try:
        for k in procs.keys():
            procs[k].start()
        
        for k in work_names:
            procs[k].join()
        procs['main'].join()
        end_time = time.time()
        print(f"elapsed time: {end_time-start_time}")
    except KeyboardInterrupt:
        end_time = time.time()
        print(f"elapsed time: {end_time-start_time}")
    

if __name__ == "__main__":
    sample_name="wikidata_0_1_10_v2_path_weight_loss"
    scale="planrgcn_binner"
    
    # for running for a specific amount of time
    #timeout -s 2 7200 python3 -m load_balance.query_balancer
    
    #main_balance_runner_v2(sample_name, scale, url = 'http://172.21.233.23:8891/sparql', bl_type='planRGCN')
    sample_name="wikidata_0_1_10_v3_path_weight_loss"
    scale="planrgcn_binner_litplan"
    url = "http://172.21.233.23:8891/sparql"
    main_balance_runner(sample_name, scale, url=url)
    #main_balance_runnerFIFO(sample_name, scale)
