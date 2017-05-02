import numpy as np
import pandas as pd
import yappi
from rut import sample, save_result

if __name__ == "__main__":
    yappi.start()
    data = pd.read_table(
        '/Users/ambrose/google_drive/manuscripts/rut/data/r_comparisons'
        '/cluster_21_vs_28_for_R.txt', index_col=0).T

    half = int(data.shape[0] / 2)
    labels = np.array((['21'] * half) + (['21_adj'] * half))
    obj = sample.Sampled(data, labels)
    yappi.start()
    result = obj.run(
        100,
        obj.mwu_map,
        obj.mwu_reduce,
        8,
    )
    save_result.save_result(result, 'test_de.csv.gz')
    with open('yappi.txt', 'w') as f:
        yappi.get_func_stats().print_all(out=f)
    with open('yappi_threads.txt', 'w') as f:
        yappi.get_thread_stats().print_all(out=f)
    print('complete!')
