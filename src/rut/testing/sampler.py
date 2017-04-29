import numpy as np
import pandas as pd
from rut import sample


if __name__ == '__main__':
    labels = np.array(([0] * 5) + ([1] * 5))
    data = pd.DataFrame(np.random.randint(0, 10, 100).reshape(10, 10))
    obj = sample.Sampled(data, labels)
    print(obj.data)
    print(obj.run(4, obj.datasum, np.sum, 4))
