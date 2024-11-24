import numpy as np
from scipy.stats import norm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import sys
from pathlib import Path

sys.path.append(str(Path(f"{__file__}").parent.parent))
from calib_functions import NLL_single, CRPS_single, NLL_bin, CRPS_bin, CRPS_weighted, get_metrics, get_metrics_base, get_F_invs

def get_sharpness_base(test_means, test_stds, p1, p2):
  return norm.ppf(p2, loc = test_means, scale = test_stds) - norm.ppf(p1, loc = test_means, scale = test_stds)

def get_sharpness(V_test, C_test, p1, p2):
  qs = get_F_invs(V_test, C_test, np.array([p2, p1]))
  return qs[:,0] - qs[:,1]

path = 'results'

Path(path + '/metrics').mkdir(parents=True, exist_ok=True)

methods = [
  NLL_single,
  CRPS_single,
  NLL_bin,
  CRPS_bin,
  CRPS_weighted
]


seed = 1
split = np.load(path + '/splits/' + str(seed) + '.npz')

valid_data, test_data = split['valid_data'], split['test_data']
valid_means, valid_stds, validy = valid_data[:,0], valid_data[:,1], valid_data[:,2]
test_means, test_stds, testy = test_data[:,0], test_data[:,1], test_data[:,2]

sharpness_arr = np.zeros((2, len(testy), len(methods) + 1))

sharpness_arr[0, :,0] = get_sharpness_base(test_means, test_stds, 0.05, 0.95)
sharpness_arr[1, :,0] = get_sharpness_base(test_means, test_stds, 0.25, 0.75)

#For each method, get the recalibrated test CDFs (given by the rows of V_test, C_test) and compute the evaluation metrics.
for method_n in range(len(methods)):
  print(methods[method_n].__name__, sep = ' ')
  V_test, C_test = methods[method_n](valid_means, valid_stds, validy, test_means, test_stds)
  sharpness_arr[0, :,method_n + 1] = get_sharpness(V_test, C_test, 0.05, 0.95)
  sharpness_arr[1, :,method_n + 1] = get_sharpness(V_test, C_test, 0.25, 0.75)
  
res_methods = ['Base', 'NLL', 'CRPS', 'NLL Bin', 'CRPS Bin', 'Weight']

sns.set_style('dark')
colors = ['cyan', 'salmon', 'lightgreen', 'plum', 'deepskyblue', 'pink']
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

  fig, ax = plt.subplots()
  ax.yaxis.grid(True) # Hide the horizontal gridlines
  ax.xaxis.grid(False)
  bplot = ax.bxp(boxes, showfliers=False, patch_artist = True,
        boxprops = { 'facecolor': 'bisque'},
  medianprops = {'color': 'black'})
  ax.set_ylim([0,5])
  for patch, color in zip(bplot['boxes'], colors):
      patch.set_facecolor(color)
  ax.set_xlabel('Method')
  ax.set_ylabel('Sharpness')
  ax.set_title('Width of ' + str(p) + '% Prediction Interval')
  return fig

ps =  [0.1, 0.25, 0.5, 0.75, 0.9]
Path('sharpness_diagrams').mkdir(exist_ok=True, parents=True)
fig = boxplot(np.quantile(sharpness_arr[0,:,:], ps, axis = 0).T, 90)
fig.savefig('sharpness_diagrams/california_sharpness90.png', dpi = 300, bbox_inches = 'tight')

fig = boxplot(np.quantile(sharpness_arr[1,:,:], ps, axis = 0).T, 50)
fig.savefig('sharpness_diagrams/california_sharpness50.png', dpi = 300, bbox_inches = 'tight')
