import numpy as np
import pandas as pd
from rut import sample
import yappi

# these tests can be tested with the current version of memory-profiler (0.46)
# using: mprof run --include-children python3 src/rut/testing/sampler.py

if __name__ == '__main__':
    # yappi.start()
    labels = np.array(([0] * 5) + ([1] * 5))
    data = pd.DataFrame(
        np.random.randint(0, 10, 100).reshape(10, 10),
        index=list('abcdefghij'),
        columns=list('klmnopqrst')
    )
    feature_sets = {'ten': [0, 1, 2, 3], 'gtf': [1, 2], 'ded': [3, 7]}
    obj = sample.Sampled(data, labels, feature_sets=feature_sets)
    print(obj.run(4, obj.kw_map, obj.kw_reduce, 4))
    # yappi.get_func_stats().print_all()
    # yappi.get_thread_stats().print_all()
