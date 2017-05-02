import numpy as np
import pandas as pd
from rut import sample
import yappi

# these tests can be tested with the current version of memory-profiler (0.46)
# using: mprof run --include-children python3 src/rut/testing/sampler.py

if __name__ == '__main__':
    # yappi.start()
    x = np.array([
        np.random.randint(0, 5, 10),  # lower than y
        np.ones(10) * 100,  # large, takes up most of library size normalization
        np.random.randint(5, 10, 10)  # higher than y
    ]).T

    y = np.array([
        np.random.randint(5, 10, 10),  # higher than y
        np.ones(10) * 100,  # takes up most of library size normalization
        np.random.randint(0, 5, 10)  # lower than y
    ]).T

    data = pd.DataFrame(
        data=np.concatenate([x, y], axis=0)
    )
    labels = np.concatenate([np.ones(10), np.zeros(10)], axis=0)

    obj = sample.Sampled(data, labels)
    print(obj.run(4, obj.kw_map, obj.kw_reduce, 4))
    # yappi.get_func_stats().print_all()
    # yappi.get_thread_stats().print_all()
