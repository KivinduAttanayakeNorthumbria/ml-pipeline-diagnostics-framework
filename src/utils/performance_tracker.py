import time
import psutil
import os

# Track time and memory
class PerformanceTracker(object):
    def __init__(self):
        self.results = []

    def start_performance_track(self):
        self.start_time = time.time()
        self.start_memory = psutil.Process(os.getpid()).memory_info().rss

    def stop_performance_track(self, operation_name):
        end_time = time.time()
        end_memory = psutil.Process(os.getpid()).memory_info().rss
        elapsed_time = end_time - self.start_time
        memory_usage = max(0, (end_memory - self.start_memory) / (1024 * 1024))

        result = {
            'operation': operation_name,
            'time_seconds': round(elapsed_time, 2),
            'memory_usage': round(memory_usage, 2),
        }

        self.results.append(result)
        print(f"{operation_name}: Time:{elapsed_time:.2f}s, Memory:{memory_usage:.2f}MB")
        return result

    def get_results(self):
        return self.results

    def clear_results(self):
        self.results = []

