#Generates Fig. 1, 2, and Table 3 in the Appendix.
import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib as mpl
from tqdm import tqdm
import sys
from pathlib import Path
sys.path.append(str(Path(f"{__file__}").parent.parent))
from calib_functions import *
mpl.rcParams['figure.dpi'] = 300
sns.set()

N = 6000
T = 12000
seeds = np.arange(10)

fontsize = 20
legend_fontsize = 17
r = 0.6
figsize1 = (12*r,6*r)
figsize2 = (7*r,6*r)
def compare_CRPS_NLL(valid_means, valid_stds, validy, valid_Fts, label):
  (valid_means, valid_stds, 
  validy, valid_Fts) = filter_valid_data(valid_means, valid_stds, validy)
  
  H,Hc = get_triangular(valid_means, valid_stds, valid_Fts)
  
  test_means = np.array([1])
  test_bin_numbers, valid_bin_numbers = np.zeros(len(test_means)).astype(int), np.zeros(len(valid_means)).astype(int)
  wts = get_bin_arr(valid_bin_numbers).astype(int).T
  wts_test = np.take(wts, test_bin_numbers, axis = 0).T
  C_CRPS = get_C_triangular(H, Hc, wts_test)
  
  bin_arr = get_bin_arr(valid_bin_numbers)
  C_NLL = np.apply_along_axis(kuleshov_helper, axis = 0, arr = bin_arr).T  
  
  plt.figure(figsize = figsize1)
  plt.plot(valid_Fts, C_CRPS[0,1:C_CRPS.shape[1]], label = 'CRPS', color = 'purple')
  plt.plot(valid_Fts, C_NLL[0,1:C_NLL.shape[1]], label = 'NLL', color = 'green')
  plt.plot(valid_Fts, valid_Fts, color = 'black', label = 'Optimal')
  plt.legend()
  plt.xticks(fontsize = fontsize)
  plt.yticks(fontsize = fontsize)
  plt.xlabel('PIT Value', fontsize = fontsize)
  plt.ylabel('New PIT Value', fontsize = fontsize)
  plt.legend(fontsize = legend_fontsize)
  plt.title('Composition Learned ' + label,fontdict = {'fontsize':fontsize})
  plt.savefig(label + "Composition.png", dpi = 300, bbox_inches = 'tight')
  plt.show()

  plt.figure(figsize = figsize2)
  plt.hist(valid_Fts, color= 'red')
  plt.title('Histogram ' + label, fontdict = {'fontsize':fontsize})
  plt.xticks(fontsize = fontsize)
  plt.yticks(fontsize = fontsize)
  plt.xlabel('PIT Values', fontsize = fontsize)
  plt.ylabel('Count', fontsize = fontsize)
  plt.savefig(label + "PITs.png", dpi = 300, bbox_inches = 'tight')
  plt.show()  

def plot_PITs(PIT_imp, PIT_corrupt_imp, cond1, cond2, label):
  PIT_imp = np.maximum(PITs, 1 - PITs)
  PIT_corrupt_imp = np.maximum(PITs_corrupt, 1 - PITs_corrupt)
  plt.figure(figsize=figsize2)
  plt.scatter(sigma[cond1], PITs[cond1], s = 1)
  plt.xlabel('Std. Dev. of Predicted CDF', fontsize = fontsize)
  plt.ylabel('PIT Value', fontsize = fontsize)
  plt.xticks(fontsize = fontsize)
  plt.yticks(fontsize = fontsize)
  plt.title('Without Outliers ' + label, fontdict = {'fontsize':fontsize})
  plt.savefig("scatter" + label + ".png", dpi = 300, bbox_inches = 'tight')
  plt.show()
  
  plt.figure(figsize=figsize2)
  plt.scatter(sigma[cond2], PITs_corrupt[cond2], s = 1, color = 'red')
  plt.xlabel('Std. Dev. of Predicted CDF', fontsize = fontsize)
  plt.ylabel('PIT Value', fontsize = fontsize)
  plt.xticks(fontsize = fontsize)
  plt.yticks(fontsize = fontsize)
  plt.title('With Outliers' + label, fontdict = {'fontsize':fontsize})
  plt.savefig("scatter_corrupt" + label + ".png", dpi = 300, bbox_inches = 'tight')
  plt.show()  

#Generate Fig. 1 and Fig. 2
np.random.seed(2)
mu = np.random.uniform(size = N)
sigma = np.random.uniform(0, 1, size = N)
valid_y = np.random.normal(mu, sigma)
PITs = norm.cdf(valid_y, loc = mu, scale = sigma)
rand_idcs = np.random.choice(np.arange(N), size = int(0.4*N),
                            replace = False)
sigma_corrupt = sigma.copy()

sigma_corrupt[rand_idcs] = 1
valid_y_corrupt = np.random.normal(mu, sigma_corrupt)
PITs_corrupt = norm.cdf(valid_y_corrupt, loc = mu, scale = sigma)

compare_CRPS_NLL(mu, sigma, valid_y_corrupt, PITs_corrupt, 'with Outliers')

valid_y_normal = np.random.normal(mu, sigma)
PITs_normal = norm.cdf(valid_y_normal, loc = mu, scale = sigma)  
compare_CRPS_NLL(mu, sigma, valid_y_normal, PITs_normal, 'without Outliers')

plot_PITs(PITs_normal, PITs_corrupt, PITs_normal > -1, PITs_corrupt > -1, "")
plot_PITs(PITs_normal, PITs_corrupt, PITs_normal < 1 - 0.99, PITs_corrupt < 1 - 0.99, "(Extreme Values)")

#Generates Table 3. 
overall_df = []
for seed in tqdm(seeds):
  np.random.seed(seed)
  mu = np.random.uniform(size = N)
  sigma = np.random.uniform(0, 1, size = N)
  valid_y = np.random.normal(mu, sigma)
  PITs = norm.cdf(valid_y, loc = mu, scale = sigma)
  rand_idcs = np.random.choice(np.arange(N), size = int(0.4*N),
                              replace = False)
  sigma_corrupt = sigma.copy()
  sigma_corrupt[rand_idcs] = 1
  valid_y_corrupt = np.random.normal(mu, sigma_corrupt)
  PITs_corrupt = norm.cdf(valid_y_corrupt, loc = mu, scale = sigma)

  test_mu = np.random.uniform(size = T)
  test_sigma = np.random.uniform(0, 1, size = T)
  test_y = np.random.normal(test_mu, test_sigma)


  V_test, C_CRPS = CRPS_single(mu, sigma, valid_y_corrupt, test_mu, test_sigma)
  res1 = get_metrics(V_test, C_CRPS, test_y)

  
  V_test, C_NLL = NLL_single(mu, sigma, valid_y_corrupt, test_mu, test_sigma)
  res2 = get_metrics(V_test, C_NLL, test_y)

  res3 = get_metrics_base(test_mu, test_sigma, test_y)

  overall = np.hstack([res1.reshape(-1,1), res2.reshape(-1,1), res3.reshape(-1,1)])
  overall_df.append(overall)

import pandas as pd
df = np.stack(overall_df)
df = pd.DataFrame(df.mean(0), columns = ['CRPS', 'NLL', 'Identity'], index = ['MSE', 'PCE', 'PBL']).to_csv('metrics.csv')

