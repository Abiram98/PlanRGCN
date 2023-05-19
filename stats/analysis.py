import sys
import configparser
import json

import numpy as np

from feature_extraction.constants import PATH_TO_CONFIG_GRAPH
from stats.utils import bxplot_w_info


def analyse_run_times():
    parser = configparser.ConfigParser()
    parser.read(PATH_TO_CONFIG_GRAPH)
    path = parser['data_files']['val_data']
    data = None
    with open(path,'rb') as f:
        data = json.load(f)
                
    if data == None:
        print('Data could not be loaded!')
        return
    with_runtimes = []
    without_runtimes = []
    for bgp, info in data.items():
        #runtime is in nanoseconds
        with_runtimes.append(info['with_runtime']/(1e9)) #nanosecond to second
        without_runtimes.append(info['without_runtime']/(1e9))
    with_runtimes,without_runtimes = np.array(with_runtimes), np.array(without_runtimes)
    print('Without BLF')
    print_val_stat(without_runtimes)
    print('With BLF')
    print_val_stat(with_runtimes)
    print('Percentage deviation')
    devias = np.array([deviation(y,x) for x,y in zip(with_runtimes,without_runtimes)])
    print(devias)
    print_val_stat(devias, plot="bxp.png")
    
def deviation(x_true,x_m):
    return (x_m - x_true)/x_true*100
def abs_diff(x_true,x_m):
    return abs(x_true - x_m)

def print_val_stat(arr, plot=None):
    print(f'\tMean: {arr.mean()}')
    print(f'\tMin: {np.min(arr)}')
    print(f'\t25%-quantile: {np.quantile(arr, q=0.25)}')
    print(f'\t50%-quantile: {np.quantile(arr, q=0.50)}')
    print(f'\t75%-quantile: {np.quantile(arr, q=0.75)}')
    print(f'\tMax: {np.max(arr)}')
    print(f'\tSTD: {np.std(arr)}')
    if plot != None:
        bxp_data = [
        {
            'label' : "Run Time Deviation Plots",
            'whislo': np.min(arr),    # Bottom whisker position
            'q1'    : np.quantile(arr, q=0.25),    # First quartile (25th percentile)
            'med'   : np.quantile(arr, q=0.5),    # Median         (50th percentile)
            'q3'    : np.quantile(arr, q=0.75),    # Third quartile (75th percentile)
            'whishi': np.max(arr),    # Top whisker position
            'fliers': []        # Outliers
        }
    
        ]
        bxplot_w_info(bxp_data,"Percentage Deviations", save=plot, y_range = [-80,50], scale='linear')

def check_absnt_pred_feats(extracted_dir='/work/data/extracted_statistics'):
    all_preds = json.load(open(f"{extracted_dir}/predicates_only.json"))
    pred_freq = json.load(open(f"{extracted_dir}/updated_pred_freq.json"))
    #lits = json.load(open(f"{extracted_dir}/updated_pred_unique_lits.json"))
    lits = json.load(open(f"{extracted_dir}/updated_nt_pred_unique_lits.json"))
    ents = json.load(open(f"{extracted_dir}/updated_nt_pred_unique_subj_obj.json"))
    print(f"# of absent predicate frequencies {len(all_preds)-len(list(pred_freq.keys()))} of {len(all_preds)}")
    print(f"# of absent predicate lits {len(all_preds)-len(list(lits.keys()))} of {len(all_preds)}")
    print(f"# of absent predicate ents {len(all_preds)-len(list(ents.keys()))} of {len(all_preds)}")
  

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == 'analysis':
        analyse_run_times()
    elif len(sys.argv) > 1 and sys.argv[1] == 'check_absnt_pred':
        check_absnt_pred_feats()