import numpy as np
import pandas as pd
import yappi
from rut import mannwhitneyu

if __name__ == "__main__":
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
    result = mannwhitneyu(x, y, n_iter=4, processes=4)
    print(result)
    # with open('yappi.txt', 'w') as f:
    #     yappi.get_func_stats().sort('tsub', sort_order='desc').print_all(out=f)
    # with open('yappi_threads.txt', 'w') as f:
    #     yappi.get_thread_stats().print_all(out=f)

