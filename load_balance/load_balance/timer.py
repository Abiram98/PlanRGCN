class Timer:
    def __init__(self):
        self.current_time = 0
        self.increment = 0.001 #1ms
        
    def step(self):
        self.current_time += self.increment
        
    def time(self):
        return self.current_time
