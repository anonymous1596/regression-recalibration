#This script generates the results in Table 1. 

import numpy as np
import pandas as pd
import os
import sys
from pathlib import Path
from time import time 

sys.path.append(str(Path(f"{__file__}").parent.parent))
from calib_functions import NLL_single, CRPS_single, NLL_bin, CRPS_bin, CRPS_weighted, get_metrics, get_metrics_base

if not(os.path.exists('results')):
  os.mkdir('results')


methods = [
  NLL_single,
  CRPS_single,
  NLL_bin,
  CRPS_bin,
  CRPS_weighted
]
seeds = np.arange(10)
times = np.zeros((len(seeds), len(methods)))
for seed_n in range(len(seeds)):
  seed = seeds[seed_n]
  split = np.load('splits/split_' + str(seed) + '.npz')
  
  valid_data, test_data = split['valid_data'], split['test_data']
  valid_means, valid_stds, validy = valid_data[:,0], valid_data[:,1], valid_data[:,2]
  test_means, test_stds, testy = test_data[:,0], test_data[:,1], test_data[:,2]
  res = [get_metrics_base(test_means, test_stds, testy)]
  
  #For each method, get the recalibrated test CDFs (given by the rows of V_test, C_test) and compute the evaluation metrics.
  for method_n in range(len(methods)):
    start = time()
    V_test, C_test = methods[method_n](valid_means, valid_stds, validy,
                                test_means, test_stds)
    end = time()
    metrics = get_metrics(V_test, C_test, testy)
    times[seed_n, method_n] =  end - start
    pd.DataFrame(times, index = np.arange(10), columns = ['NLL', 'CRPS', 'NLL Bin', 'CRPS Bin', 'Weight']).to_csv('times.csv')
    res.append(metrics)
  df = pd.DataFrame(np.stack(res), 
               columns = ['MSE', 'PCE', 'PBL'], 
               index = ['Base', 'NLL', 'CRPS', 'NLL Bin', 'CRPS Bin', 'Weight'])
  df.to_csv('results/' + str(seed) + '.csv')

res = []  
for seed in seeds:
  res.append(pd.read_csv('results/' + str(seed) + '.csv', index_col = 0).to_numpy())
res = pd.DataFrame(np.stack(res).mean(0), 
                   columns = ['MSE', 'PCE', 'PBL'], 
                   index = ['Base', 'NLL', 'CRPS', 'NLL Bin', 'CRPS Bin', 'Weight'])
res.to_csv('results/average_results.csv')
