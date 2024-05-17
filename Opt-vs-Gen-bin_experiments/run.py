# This script generates Fig. 3 and Fig. 4 in the Appendix.
import torch
import numpy as np
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
sns.set()
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
from scipy.stats import norm
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(f"{__file__}").parent.parent))
from calib_functions import *

seeds = np.arange(10)


np.random.seed(1)
n_intervals = 10
randidcs = np.random.permutation(n_intervals)
unif_ps = np.random.uniform(0,1,3000)
n_bins_arr = np.arange(1,21,1)
MSEs = np.zeros((len(n_bins_arr),2,len(seeds)))
PCEs = np.zeros((len(n_bins_arr),2, len(seeds)))
PBLs = np.zeros((len(n_bins_arr),2, len(seeds)))
rand_stds = np.linspace(-0.3,0.3, n_intervals)

rand_stds = rand_stds[randidcs]
intervals = np.linspace(0,1,n_intervals+1)
xs = np.arange(0,1,0.01)

fontsize = 18
plt.plot(xs, xs, color = 'blue', label = 'Forecasted')
plt.plot(xs, xs + rand_stds[get_bin_number(xs, intervals)], color = 'red', label = 'True')
plt.legend(fontsize=fontsize)
plt.xticks(fontsize = fontsize)
plt.yticks(fontsize = fontsize)
plt.savefig('bin_mean.png', dpi = 300, bbox_inches= 'tight')

for seed_n in range(len(seeds)):
  seed = seeds[seed_n]

  torch.manual_seed(seed)
  np.random.seed(seed)

  N = 500  
  mu = np.random.uniform(size = N)
  sigma = np.random.uniform(0, 1, size = N)
  valid_bin_numbers = get_bin_number(mu, intervals)
  mults = rand_stds[valid_bin_numbers]
  new_mu = mu + mults
  validy = np.random.normal(new_mu, sigma)
  
  def get_place(n, place):
    return str(n)[-1]
    
  valid_means = mu
  valid_stds = sigma
  
  T = 6000
  mu_test = np.random.uniform(size = T)
  sigma_test = np.random.uniform(0, 1, size = T)
  test_bin_numbers = get_bin_number(mu_test, intervals)
  test_mults = rand_stds[test_bin_numbers]

  new_mu_test = mu_test + test_mults
  testy = np.random.normal(new_mu_test, sigma_test)
  
  test_means = mu_test 
  test_stds = sigma_test
  
  ps = np.arange(0.05, 1, 0.05)
  
  for j in tqdm(range(len(n_bins_arr))):
    n_bins = n_bins_arr[j]
    (valid_means, valid_stds, 
    validy, valid_Fts) = filter_valid_data(valid_means, valid_stds, validy)
    (test_bin_numbers, 
    valid_bin_numbers) = get_test_valid_bin_numbers(test_means, valid_means, n_bins)        
    
    #Optimization Binning Approach
    wts = get_bin_arr(valid_bin_numbers).astype(int).T
    wts_test = np.take(wts, test_bin_numbers, axis = 0).T
    H,Hc = get_triangular(valid_means, valid_stds, valid_Fts)
    C_opt = get_C_triangular(H, Hc, wts_test)
    
    V_opt = get_V(valid_Fts, test_means, test_stds)  
    metrics = get_metrics(V_opt, C_opt, testy)
    
    
    MSEs[j,0, seed_n] = metrics[0]
    PCEs[j,0,seed_n] = metrics[1]
    PBLs[j,0,seed_n] = metrics[2]

      
    #General Binning Approach
    val_bin_arr = get_bin_arr(valid_bin_numbers)
    test_bin_arr = get_bin_arr(test_bin_numbers)
    
    C_gen = np.zeros(C_opt.shape)
    
    #These empty arrays for the predicted means and quantiles of the recalibrated test CDFs will be filled in one bin at a time.
    gen_test_means = np.zeros(len(testy))
    gen_test_F_invs = np.zeros((len(testy), len(ps)))
    
    for i in range(n_bins):
      #For each bin, obtain the subset of observations in that bin.
      valid_means_bin = valid_means[val_bin_arr[:,i]]
      valid_stds_bin = valid_stds[val_bin_arr[:,i]]
      validy_bin = validy[val_bin_arr[:,i]]
      valid_Fts_bin = valid_Fts[val_bin_arr[:,i]]
      
      
      testy_bin = testy[test_bin_arr[:,i]]
      test_means_bin = test_means[test_bin_arr[:,i]]
      test_stds_bin = test_stds[test_bin_arr[:,i]]
      
      V_bin, C_bin = CRPS_single(valid_means_bin, valid_stds_bin, validy_bin, test_means_bin, test_stds_bin)
      gen_test_means[test_bin_arr[:,i]] = get_means(V_bin, C_bin)
      gen_test_F_invs[test_bin_arr[:,i],:] = get_F_invs(V_bin, C_bin, ps)

    testy_tile = np.tile(testy.reshape(-1,1),(1, len(ps)))
    ps_tile = np.tile(np.expand_dims(ps,0), (len(testy), 1))
    MSEs[j,1,seed_n] =  ((gen_test_means - testy) ** 2).mean()
    PCEs[j,1,seed_n] = np.square((testy_tile < gen_test_F_invs).mean(0)-ps).mean()
    PBLs[j,1,seed_n] = (
      (gen_test_F_invs > testy_tile)*(gen_test_F_invs - testy_tile)*(1 - ps_tile) + 
      (gen_test_F_invs < testy_tile)*(testy_tile - gen_test_F_invs)*(ps_tile) 
    ).mean()


def plot(opt_results, gen_results, label):
  xs = n_bins_arr.astype(int)
  r = 1
  figsize = (12*r,3*r)
  fontsize = 18
  plt.figure(figsize=figsize)
  plt.plot(xs, opt_results, label = 'Optimization Binning')
  plt.scatter(xs, opt_results, s = 10)
  plt.plot(xs, gen_results, label = 'General Binning')
  plt.scatter(xs, gen_results, s = 10)
  plt.legend(fontsize = fontsize)
  plt.xlabel("Number of Bins", fontsize = fontsize)
  plt.xticks(np.arange(2,22,2), fontsize = fontsize)
  plt.yticks(fontsize = fontsize)
  plt.ylabel(label, fontsize = fontsize)
  plt.savefig(label + ".png", bbox_inches = 'tight')

plot(np.mean(MSEs[:,0,:], axis = 1), np.mean(MSEs[:,1,:], axis = 1), 'MSE')
plot(np.mean(PCEs[:,0,:], axis = 1), np.mean(PCEs[:,1,:], axis = 1), 'PCE')
plot(np.mean(PBLs[:,0,:], axis = 1), np.mean(PBLs[:,1,:], axis = 1), 'PBL')