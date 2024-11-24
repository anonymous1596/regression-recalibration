import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from time import time
import sys
from pathlib import Path

sys.path.append(str(Path(f"{__file__}").parent.parent))
from calib_functions import CRPS_weighted, get_metrics_base, get_metrics, get_F_invs

def get_sharpness_base(test_means, test_stds, p1, p2):
  return norm.ppf(p2, loc = test_means, scale = test_stds) - norm.ppf(p1, loc = test_means, scale = test_stds)

def get_sharpness(V_test, C_test, p1, p2):
  qs = get_F_invs(V_test, C_test, np.array([p2, p1]))
  return qs[:,0] - qs[:,1]

test_years = [1991, 1992, 1993, 1994, 1995]

dat = np.load('dat.npy')

learn_nbins = False
times = []
res_base = []
res_recalib = []
iter_n = 0
test_year = 1993
month = 12
optimal_n_bins = np.load('pinatubo_learned_nbins.npy')[28]
n_jobs = 4
  
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

sharpness_base90 = get_sharpness_base(test_means, test_stds, 0.05, 0.95)
sharpness_base50 = get_sharpness_base(test_means, test_stds, 0.25, 0.75)

V_test, C_test = CRPS_weighted(valid_means, valid_stds, validy, test_means, test_stds, optimal_n_bins, n_jobs)

sharpness_weight90 = get_sharpness(V_test, C_test, 0.05, 0.95)
sharpness_weight50 = get_sharpness(V_test, C_test, 0.25, 0.75)


res_methods = ['Base', 'Weight']

sns.set_style('dark')
colors = ['cyan', 'salmon']
def boxplot(quantiles, p):
  boxes = []
  for i in range(len(res_methods)):
    box = {
      'label': res_methods[i],
      'whislo': quantiles[i,0],
      'q1': quantiles[i,1],
      'med': quantiles[i,2],
      'q3': quantiles[i,3],
      'whishi': quantiles[i,4],
      'fliers': []
    }
    boxes.append(box)

  fig, ax = plt.subplots(figsize = (3,4))
  ax.yaxis.grid(True) # Hide the horizontal gridlines
  ax.xaxis.grid(False)
  bplot = ax.bxp(boxes, showfliers=False, patch_artist = True,
        boxprops = { 'facecolor': 'bisque'},
  medianprops = {'color': 'black'}, widths = 0.25)
  ax.set_ylim([0,3.8])
  for patch, color in zip(bplot['boxes'], colors):
      patch.set_facecolor(color)
  ax.set_xlabel('Method')
  ax.set_ylabel('Sharpness')
  ax.set_title('Width of ' + str(p) + '% Prediction Interval')
  return fig

ps =  [0.1, 0.25, 0.5, 0.75, 0.9]
Path('sharpness_diagrams').mkdir(exist_ok=True, parents=True)
sharpness90 = np.hstack([sharpness_base90.reshape(-1,1), 
                         sharpness_weight90.reshape(-1,1)])
fig = boxplot(np.quantile(sharpness90, ps, axis = 0).T, 90)
fig.savefig('sharpness_diagrams/90.png', dpi = 300, bbox_inches = 'tight')

sharpness50 = np.hstack([sharpness_base50.reshape(-1,1), 
                         sharpness_weight50.reshape(-1,1)])
fig = boxplot(np.quantile(sharpness50, ps, axis = 0).T, 50)
fig.savefig('sharpness_diagrams/50.png', dpi = 300, bbox_inches = 'tight')