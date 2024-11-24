#This script generates the results in Table 1. 

import numpy as np
import pandas as pd
import sys
from pathlib import Path
from time import time 

sys.path.append(str(Path(f"{__file__}").parent.parent))
from calib_functions import NLL_single, CRPS_single, NLL_bin, CRPS_bin, CRPS_weighted, get_metrics, get_metrics_base


def run_california(n_jobs):
  path = 'results'

  Path(path + '/metrics').mkdir(parents=True, exist_ok=True)

  methods = [
    NLL_single,
    CRPS_single,
    NLL_bin,
    CRPS_bin,
    CRPS_weighted
  ]
  seeds = np.arange(10)

  res_columns = ['MSE', 'PCE', 'PBL', 'CRPS']
  res_methods = ['NLL', 'CRPS', 'NLL Bin', 'CRPS Bin', 'Weight']

  times = np.zeros((len(seeds), len(methods)))

  for seed_n in range(len(seeds)):
    print('Seed: ' + str(seed_n))
    seed = seeds[seed_n]
    split = np.load(path + '/splits/' + str(seed) + '.npz')
    
    valid_data, test_data = split['valid_data'], split['test_data']
    valid_means, valid_stds, validy = valid_data[:,0], valid_data[:,1], valid_data[:,2]
    test_means, test_stds, testy = test_data[:,0], test_data[:,1], test_data[:,2]
    res = [get_metrics_base(test_means, test_stds, testy)]
    
    #For each method, get the recalibrated test CDFs (given by the rows of V_test, C_test) and compute the evaluation metrics.
    for method_n in range(len(methods)):
      if methods[method_n].__name__ in ['NLL_bin', 'CRPS_bin', 'CRPS_weighted']:
        start = time()
        V_test, C_test = methods[method_n](valid_means, valid_stds, validy,
                                    test_means, test_stds, n_jobs = n_jobs)
        end = time()
      else:
        start = time()
        V_test, C_test = methods[method_n](valid_means, valid_stds, validy,
                                    test_means, test_stds)
        end = time()
      metrics = get_metrics(V_test, C_test, testy)
      times[seed_n, method_n] =  end - start
      pd.DataFrame(times, index = seeds, columns = res_methods).to_csv('times.csv')
      res.append(metrics)
      
    df = pd.DataFrame(np.stack(res), 
                columns = res_columns, 
                index = ['Base'] + res_methods)
    df.to_csv(path + '/metrics/' + str(seed) + '.csv')

  res = []  
  for seed in seeds:
    res.append(pd.read_csv(path + '/metrics/' + str(seed) + '.csv', index_col = 0).to_numpy())
  res = np.stack(res)
  res = pd.DataFrame(np.stack(res).mean(0), 
                    columns = res_columns, 
                    index = ['Base'] + res_methods)
  res.to_csv(path + '/metrics/average_results.csv')

