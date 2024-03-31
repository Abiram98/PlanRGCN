from pathlib import Path
import load_balance.fifo_balancer as fifo
import load_balance.query_balancer as qbl
import configparser
import sys
import os
import numpy as np
import random
import pandas as pd
from  load_balance.workload.workload import Workload
from load_balance.workload.arrival_time import ArrivalRateDecider
import load_balance.const as const
import pickle

def get_workload(sample_name, scale, add_lsq_url,cls_field, mu = const.MU):
    np.random.seed(42)
    random.seed(42)
    # Workload Setup
    df = pd.read_csv(f'/data/{sample_name}/test_sampled.tsv', sep='\t')
    print(df)
    print(df['mean_latency'].quantile(q=0.25))
    w = Workload(true_field_name=cls_field)
    w.load_queries(f'/data/{sample_name}/test_sampled.tsv')
    w.set_time_cls(f"/data/{sample_name}/{scale}/test_pred.csv",add_lsq_url=add_lsq_url)
    a = ArrivalRateDecider(seed=21)
    w.shuffle_queries()
    w.shuffle_queries()
    w.set_arrival_times(a.assign_arrival_rate(w, mu= mu))
    return w
    
if __name__ == "__main__":
    #timeout -s 2 7200 python3 -m load_balance.main_balancer /path/to/config
    config = configparser.ConfigParser()
    config.read(sys.argv[1])
    sample_name=config['DATASET']['sample_name']
    scale=config['DATASET']['scale']
    url = config['DATABASE']['url']
    save_dir = config['DATASET']['save_dir']
    cls_field = config['DATASET']['true_field_name']
    std_file = config['DATASET']['stdout']
    add_lsq_url = config['DATASET'].getboolean('add_lsq_url')
    MU = config['DATASET'].getint('MU')
    os.makedirs(Path(save_dir), exist_ok=True)
    w = get_workload(sample_name, scale, add_lsq_url, cls_field, mu = MU)
    with open(os.path.join(save_dir,"workload.pck"), 'wb') as wf:
        w.pickle(wf)
    with open(os.path.join(save_dir, std_file), 'w') as sys.stdout:
        match config['TASK']['taskName']:
            case "fifo":
                workers = int(config['LOADBALANCER']['FIFOWorkers'])
                fifo.main_balance_runner(w, url=url, save_dir=save_dir, n_workers=workers)
            case "qpp":
                fast_workers = int(config['LOADBALANCER']['FastWorkers'])
                med_workers = int(config['LOADBALANCER']['MediumWorkers'])
                slow_workers = int(config['LOADBALANCER']['SlowWorkers'])
                qbl.main_balance_runner(w, url=url, save_dir=save_dir, work_dict={
                    'fast': fast_workers,
                    'med' : med_workers,
                    'slow': slow_workers
                })

