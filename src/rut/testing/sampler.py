import numpy as np
import pandas as pd
from rut import sample

# these tests can be tested with the current version of memory-profiler (0.46)
# using: mprof run --include-children python3 src/rut/testing/sampler.py

if __name__ == '__main__':
    labels = np.array(([0] * 5) + ([1] * 5))
    data = pd.DataFrame(np.random.randint(0, 10, 100).reshape(10, 10))
    obj = sample.Sampled(data, labels)
    print(obj.data)
    print(obj.run(4, obj.datasum, lambda x: x, 4))
