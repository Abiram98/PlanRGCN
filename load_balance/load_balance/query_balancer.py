import load_balance.arrival_time
from load_balance.arrival_time import ArrivalRateDecider
import pandas as pd
from  load_balance.workload import Workload
import copy

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
if __name__ == "__main__":
    sample_name="wikidata_0_1_10_v2_path_weight_loss"
    scale="planrgcn_binner"
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
    