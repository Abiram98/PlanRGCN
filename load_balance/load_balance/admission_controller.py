from pathlib import Path
import load_balance.fifo_balancer as fifo
import load_balance.query_balancer as qbl
import configparser, argparse
import sys
import os
import numpy as np
import random
import pandas as pd
from  load_balance.workload.workload import Workload
from load_balance.workload.arrival_time import ArrivalRateDecider
import load_balance.const as const


def get_workload(query_file, predicted_file, add_lsq_url,cls_field, mu = const.MU, seed=42):
    np.random.seed(seed)
    random.seed(seed)
    # Workload Setup
    #df = pd.read_csv(query_file, sep='\t')
    #print(df)
    #print(df['mean_latency'].quantile(q=0.25))
    w = Workload(true_field_name=cls_field)
    w.load_queries(query_file)
    #print('after load', df)
    w.set_time_cls(predicted_file,add_lsq_url=add_lsq_url)
    a = ArrivalRateDecider(seed=seed)
    w.shuffle_queries()
    w.shuffle_queries()
    w.set_arrival_times(a.assign_arrival_rate(w, mu= mu))
    return w


if __name__ == "__main__":
    # timeout -s 2 7200 python3 -m load_balance.main_balancer qpp -d wikidata_0_1_10_v3_path_weight_loss -b planrgcn_binner_litplan -t planrgcn_prediction -o /tmp/test -r 44 -u http://172.21.233.14:8891/sparql -f 4 -m 4 -s 2 -i 10 --seed 42

    parser = argparse.ArgumentParser(
        prog='Admission Controller',
        description='Admission Control on test dataset workload',
        epilog='')

    parser.add_argument('-f', '--query_file')
    parser.add_argument('-p', '--prediction_file')
    parser.add_argument('-t', '--true_field_name')
    parser.add_argument('-o', '--save_dir')
    parser.add_argument('-l', '--add_lsq_url', default='yes')
    parser.add_argument('-r', '--MU', default=44, type=int)
    parser.add_argument('-u', '--url')
    parser.add_argument('-i', '--FIFOWorkers')
    parser.add_argument('--interval', default=2, type=int)
    parser.add_argument('--seed', default=42, type=int)
    args = parser.parse_args()
    query_file = args.query_file
    predicted_file = args.prediction_file
    url = args.url
    save_dir = args.save_dir
    cls_field = args.true_field_name
    std_file = "main_file.log"
    add_lsq_url = True if args.add_lsq_url.lower() == 'yes' else False
    MU = int(args.MU)

    os.makedirs(Path(save_dir), exist_ok=True)
    print('here')

    w = get_workload(query_file, predicted_file, add_lsq_url, cls_field, mu=MU, seed=args.seed)
    with open(os.path.join(save_dir, "workload.pck"), 'wb') as wf:
        w.pickle(wf)
    print(args)
    with open(os.path.join(save_dir, std_file), 'w') as sys.stdout:
        workers = int(args.FIFOWorkers)
        fifo.main_admission_runner(w, url=url, save_dir=save_dir, n_workers=workers)