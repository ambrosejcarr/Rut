import numpy as np
import pandas as pd
from rut import sample
import yappi

# these tests can be tested with the current version of memory-profiler (0.46)
# using: mprof run --include-children python3 src/rut/testing/sampler.py

if __name__ == '__main__':
    # yappi.start()
    n = 100
    labels = np.array(([0] * int(n/2)) + ([1] * int(n/2)))
    data = pd.DataFrame(np.random.randint(0, 10, 10000).reshape(100, 100))
    obj = sample.Sampled(data, labels)
    print(obj.run(4, obj.cluster_map, obj.cluster_reduce, 4))
    # yappi.get_func_stats().print_all()
    # yappi.get_thread_stats().print_all()
