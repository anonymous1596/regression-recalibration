#This script produces the data in Fig. 6.
import numpy as np
import pandas as pd
from time import time
import sys
from pathlib import Path

sys.path.append(str(Path(f"{__file__}").parent.parent))
from calib_functions import CRPS_weighted, get_metrics_base, get_metrics


def run_pinatubo(n_jobs, learn_nbins):
  test_years = [1991, 1992, 1993, 1994, 1995]

  #Run the script get_pinatubo_functions.py to obtain dat.npy, which contains the CNN's predictions.
  dat = np.load('dat.npy')

  #Set learn_bins = True to learn the number of bins in each experiment.
  #If learn_bins = True, set n_jobs for the number of cores to use


  times = []
  res_base = []
  res_recalib = []
  iter_n = 0
  for test_year in test_years:
    #This code creates the validation and test dataset, where we use up to one year of the previous data before the eruption for validation.  
    if test_year == 1991:
      months = np.arange(8,13,1)
    else:
      months = np.arange(1,13,1)
    for month in months:
      
      if learn_nbins:
        optimal_n_bins = None
      else:
        optimal_n_bins = np.load('pinatubo_learned_nbins.npy')[iter_n]
      iter_n += 1
        
      print(str(test_year) + ' ' + str(month), sep = ' ')
      cond1 = (dat[:,4] == test_year) & (dat[:,3] == month)
      if test_year == 1991:
        cond2 = (dat[:,4] == 1991) & (dat[:,3] <= month - 1) & (dat[:,3] >= 7)
      elif test_year == 1992:
        if month <= 7:
          cond2 = ((dat[:,4] == 1991) & (dat[:,3] >= 7)) | ((dat[:,4] == 1992) & (dat[:,3] <= month - 1))
        else:
          cond2 = ((dat[:,4] == 1991) & (dat[:,3] >= month)) | ((dat[:,4] == 1992) & (dat[:,3] <= month - 1))
      else:
        cond2 = ((dat[:,4] == test_year - 1) & (dat[:,3] >= month)) | ((dat[:,4] == test_year) & (dat[:,3] <= month - 1))
      test_data = dat[cond1]
      valid_data = dat[cond2]
    
      test_means, test_stds, testy = test_data[:,0], test_data[:,1], test_data[:,2]
      valid_means, valid_stds, validy = valid_data[:,0], valid_data[:,1], valid_data[:,2]

      start = time()
      V_test, C_test = CRPS_weighted(valid_means, valid_stds, validy, test_means, test_stds, optimal_n_bins, n_jobs)
      end = time()
      
      index = pd.MultiIndex.from_tuples([(test_year, month)], names = ['Year', 'Month'])
      
      times_df = pd.DataFrame([end - start]).T
      times_df.index = index
      times.append(times_df)
      pd.concat(times, axis = 0).to_csv('times.csv')
      
      metrics_df = pd.DataFrame(get_metrics(V_test, C_test, testy)).T
      metrics_df.index = index
      res_recalib.append(pd.DataFrame(metrics_df))
      pd.concat(res_recalib, axis = 0).to_csv('results_recalib.csv')
      
      base_df = pd.DataFrame(get_metrics_base(test_means, test_stds, testy)).T
      base_df.index = index
      res_base.append(pd.DataFrame(base_df))
      pd.concat(res_base, axis = 0).to_csv('results_base.csv')
    