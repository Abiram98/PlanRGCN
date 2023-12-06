from load_balance.query import Query
class Worker:
    def __init__(self):
        self.past_queries:list[Query] = []
        self.processing = None

    def is_free(self):
        if self.processing is None:
            return True
        return False

    #perform a step
    def step(self, time):
        if not self.is_free():
            processed_time = self.processing.starttime + self.processing.execution_time
            if processed_time >= time:
                self.processing.finish_time = time
                self.past_queries.append(self.processing)
                self.processing = None
        
    def assign_query(self, query, time):
        if self.is_free():
            query.starttime = time
            self.processing = query
            return True
        return False

