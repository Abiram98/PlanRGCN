import json


class Query:
    def __init__(self, ID, queryString, execution_time) -> None:
        self.ID = ID
        self.query_string = queryString
        self.execution_time = execution_time
        self.time_cls = 0 #0 for fast, 1 for medium, 2 for slow
        self.true_time_cls = 0 #0 for fast, 1 for medium, 2 for slow
        
        
        # Scheduling dependant info
        self.arrivaltime = None #arrival time in load balancer
        self.starttime = None #start time at worker
        self.finish_time = None #finish time at worker
    
    def set_time_cls(self, time_cls):
        self.time_cls = time_cls
        
    def set_true_time_cls(self, time_cls):
        self.true_time_cls = time_cls
    
    def __str__(self):
        dct = {'ID': self.ID, 'text': self.query_string}
        return json.dumps(dct)