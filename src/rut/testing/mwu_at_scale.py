import numpy as np
import pandas as pd
import yappi
from rut import mannwhitneyu, save_result

if __name__ == "__main__":
    data = pd.read_table(
        '/Users/ambrose/google_drive/manuscripts/rut/data/r_comparisons'
        '/cluster_21_vs_28_for_R.txt', index_col=0).T

    half = int(data.shape[0] / 2)
    a = data.iloc[:half, :]
    b = data.iloc[half:, :]

    yappi.start()
    result = mannwhitneyu(a, b, n_iter=4, processes=4)
    save_result(result, 'test_de_master.csv')
    with open('yappi_master.txt', 'w') as f:
        yappi.get_func_stats().sort('tsub', sort_order='desc').print_all(out=f)
    with open('yappi_master_threads.txt', 'w') as f:
        yappi.get_thread_stats().print_all(out=f)
    print('complete!')
