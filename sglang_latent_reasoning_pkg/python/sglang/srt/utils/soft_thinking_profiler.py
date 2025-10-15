import time
import torch

# Global dictionary to accumulate timings (milliseconds)
timers = {}

class time_section:
    """Context manager for timing sections (supports CUDA and CPU)."""
    def __init__(self, name: str):
        self.name = name
    def __enter__(self):
        if torch.cuda.is_available():
            self.start = torch.cuda.Event(enable_timing=True)
            self.end = torch.cuda.Event(enable_timing=True)
            self.start.record()
        else:
            self.start = time.perf_counter()
    def __exit__(self, exc_type, exc_val, exc_tb):
        if isinstance(self.start, torch.cuda.Event):
            self.end.record()
            torch.cuda.synchronize()
            elapsed = self.start.elapsed_time(self.end)
        else:
            elapsed = (time.perf_counter() - self.start) * 1000.0
        timers[self.name] = timers.get(self.name, 0.0) + elapsed


def print_profile():
    """Print accumulated timing results sorted by descending time."""
    if not timers:
        print("No soft thinking profiling data.")
        return
    print("\nSoft Thinking Profiling Results (ms):")
    for name, ms in sorted(timers.items(), key=lambda x: -x[1]):
        print(f"  {name}: {ms:.3f} ms") 