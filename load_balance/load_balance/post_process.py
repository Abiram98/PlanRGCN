import os
import json
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def extract_qs(path, verbose =False):
    qs = []
    timed_out_n_slow = []
    for p in os.listdir(path):
        if p == "main.json":
            continue
        if verbose:
            print(f"{path}/{p}")
        try:
            data = json.load(open(f"{path}/{p}",'r'))
            for q in data:
                q_data = json.loads(q['query'])
                for k in q_data.keys():
                    q[k] = q_data[k]
                
                if (not 'slow' in p) and q['response'] == 'timed out':
                    timed_out_n_slow.append(q)
                q['ex_time'] = q['query_execution_end'] - q['query_execution_start']
                q['latency'] = q['query_execution_end'] - q['arrival_time']
                q['queue_wait_time'] = q['query_execution_start'] - q['queue_arrival_time']
                q['arrival_time'] -= q['start_time'] 
                q['queue_arrival_time'] -= q['start_time'] 
                q['query_execution_start'] -= q['start_time'] 
                q['query_execution_end'] -= q['start_time'] 
                q['start_time'] = 0
                q['process'] = p
                qs.append(q)
        except Exception as e:
            if verbose:
                print(f"Did not work {path}/{p}: {e}")
    return qs, timed_out_n_slow

def plot_box_latency(dct, figsize=(4,6)):
    fig, ax = plt.subplots(layout='constrained', figsize=figsize)
    for k in dct.keys():
        ax = sns.boxplot(y=[q['latency'] for q in dct[k]],x=[k for q in dct[k]], ax=ax)
    #ax = sns.boxplot(y=[q['latency'] for q in qpp_lb],x=['PlanRGCN Load Balancer' for q in qpp_lb], ax=ax)
    ax.set_ylabel('Query Latency (s)')
    ax.set_title("Query Latency Plots")
    ax.tick_params(axis='x', rotation=15)
    plt.show()

def get_overview_table(path_data):
    d_p = {'Good Queries':{}, 'Time out': {}}
    for k in path_data:
        data, timeouts = extract_qs(path_data[k])
        d_p['Time out'][k] = np.sum([1 for x in data if x['ex_time'] >=900])
        d_p['Good Queries'][k] = np.sum([1 for x in data if x['response'] == 'ok'])
    df = pd.DataFrame.from_dict(d_p)   
    return df

def get_time_outs(data):
    print("Time outs")
    dct = {}
    for k in data.keys():
        val = np.sum([1 for x in data[k] if x['ex_time'] >=900])
        print(f"{k} Timeouts : {val}")
        dct[k] = val
    return dct

def get_good_qs(data):
    print("Good Queries")
    dct = {}
    for k in data.keys():
        val = np.sum([1 for x in data[k] if x['response'] == 'ok'])
        print(f"{k} Good Queries : {val}")
        dct[k] = val
    return dct
def calculate_total_latency(qs):
    sum = 0
    for q in qs:
        sum += q['latency']
    return sum
    
def calculate_avg_latency(data):
    dct = {}
    for k in data.keys():
        val = calculate_total_latency(data[k])/len(data[k])
        print(f"{k}:  {val}")
        dct[k] = val
    return dct

def plot_box_ex_time(dct, figsize=(4,4)):
    fig, ax = plt.subplots(layout='constrained', figsize=figsize)
    for k in dct.keys():
        ax = sns.boxplot(y=[q['ex_time'] for q in dct[k]],x=[k for q in dct[k]], ax=ax)
    ax.tick_params(axis='x', rotation=15)
    plt.show()

def plot_box_queu_wait_time(dct, figsize=(4,6)):
    fig, ax = plt.subplots(layout='constrained', figsize=figsize)
    for k in dct.keys():
        ax = sns.boxplot(y=[q['queue_wait_time'] for q in dct[k]],x=[k for q in dct[k]], ax=ax)
    ax.set_ylabel('Queue Wait Time (s)')
    ax.set_title("Queue Wait Time Plots (Wikidata Full)")
    ax.tick_params(axis='x', rotation=10)
    plt.show()


def plot_box_queu_wait_time_int(dct, figsize=(4,6)):
    load_balancer = []
    que_w_time = []
    runtime_interval = []
    for k in dct.keys():
        for q in dct[k]:
            load_balancer.append(k)
            que_w_time.append(q['queue_wait_time'])
            match q['true_interval']:
                case '0':
                    runtime_interval.append('fast')
                case '1':
                    runtime_interval.append('medium')
                case '2':
                    runtime_interval.append('slow')
    
    df = pd.DataFrame.from_dict({'Approach': load_balancer, 'Runtime Interval':runtime_interval, 'queue wait time':que_w_time})
    figsize=(10,6)
    fig, ax = plt.subplots(layout='constrained', figsize=figsize)
    ax.tick_params(axis='x', rotation=0)
    ax.tick_params(axis='x', labelsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax = sns.boxplot(x='Runtime Interval',y='queue wait time',data=df,hue='Approach', ax =ax, order=['fast', 'medium', 'slow'])
    ax.set_xlabel("")
    ax.set_title('Queue Wait Time by Time Intervals', fontsize=14)
    ax.set_ylabel("Queue Wait Time (s)", fontsize=14)
    ax.legend(loc=2)