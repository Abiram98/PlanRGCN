import pandas as pd
import numpy as np
from load_balance.query import Query
import random

class Workload:
    def __init__(self) -> None:
        self.queries:list[Query] = list()
        self.arrival_times:list[float] = list()
        self.p_idx = 0
        self.current_idx = 0
        random.seed(42)
    
    def queries_finished_before_other(self):
        prev, prev_arr, count = None,None, 0
        for q, t in zip(self.queries, self.arrival_times):
            if prev is not None:
                if t > (prev.execution_time + prev_arr):
                    count += 1
            prev = q
            prev_arr=t
        return count
    
    def set_arrival_times(self, arrival_times):
        if not isinstance(arrival_times, list):
            arrival_times = arrival_times.tolist()
        self.arrival_times = arrival_times
    
    def __len__(self):
        return len(self.queries)
    
    def __getitem__(self, item):
        if len(self.queries) == len(self.arrival_times):
            return (self.queries[item], self.arrival_times[item])
        return self.queries[item]
    
    def add(self, ID, text, rs):
        q = Query(ID,text,rs)
        self.queries.append(q)
    
    def pop(self):
        if len(self) == 0:
            return None,None
        q = self.queries.pop()
        a = self.arrival_times.pop()
        return q,a 
    def shuffle_queries(self):
        random.shuffle(self.queries)
        
    def set_time_cls(self, path):
        df2 = pd.read_csv(path)
        df2['id'] = df2['id'].apply(lambda x: f"http://lsq.aksw.org/{x}")
        df2 = df2.set_index('id')
        for q in self.queries:
            q.set_time_cls(df2.loc[q.ID]['planrgcn_prediction'])
            q.set_true_time_cls(df2.loc[q.ID]['time_cls'])
    
    def reorder_queries(self):
        slow_qs = []
        medium_qs = []
        fast_qs = []
        for q in self.queries:
            match q.true_time_cls:
                case 0:
                    fast_qs.append(q)
                case 1:
                    medium_qs.append(q)
                case 2:
                    slow_qs.append(q)
        for i in range(10):
            np.random.shuffle(slow_qs)
            np.random.shuffle(medium_qs)
            np.random.shuffle(fast_qs)
        m_rate = len(medium_qs)/len(fast_qs)
        s_rate = len(slow_qs)/len(fast_qs)
        print(len(slow_qs), len(medium_qs),len(fast_qs))
        
        
        
    
    def load_queries(self, path, sep='\t'):
        df = pd.read_csv(path, sep=sep)
        count = 0
        for idx, row in df.iterrows():
            try:
                self.add(row['id'],row['queryString'],row['mean_latency'])
                count += 1
            except Exception as e:
                print(f"loading of query did not work!")
        print(f"loaded {count}")