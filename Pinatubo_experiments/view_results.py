import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 300
res_base = pd.read_csv('results_base.csv')

res_recalib = pd.read_csv('results_recalib.csv')

def get_year_month(res):
  r = res.reset_index().drop(['index'], axis = 1)
  r['Date'] = r['Month'].astype(str) + '-' + r['Year'].astype(str).str[2:4]
  return r.drop(['Year', 'Month'], axis = 1)

res_base = get_year_month(res_base)
res_recalib = get_year_month(res_recalib)
xtick_fontsize = 20
legend_font_size = 20
ytick_font_size = 20
dim = (25,3.5)
def plot(date, base, recalib, label):
  plt.figure(figsize=dim)
  plt.plot(date, base, label = 'Base ' + label, color = 'red')
  plt.scatter(date, base, s = 10, color = 'red')
  plt.plot(date, recalib, label = 'Recalib. ' + label, color = 'blue')
  plt.scatter(date, recalib, s = 10, color = 'blue')
  plt.legend(fontsize = legend_font_size)
  plt.xticks(rotation = 90, fontsize = xtick_fontsize);
  plt.yticks(fontsize = ytick_font_size)
  plt.xlim([-1,len(date)])
  plt.axvline(x=4.5, color = 'black', linestyle = 'dotted')
  plt.axvline(x=16.5, color = 'black', linestyle = 'dotted')
  plt.axvline(x=28.5, color = 'black', linestyle = 'dotted')
  plt.axvline(x=40.5, color = 'black', linestyle = 'dotted')
  plt.savefig(label + ".png", bbox_inches="tight", dpi = 600)
  plt.show()  

plot(res_base['Date'], res_base['0'], res_recalib['0'], 'MSE')
plot(res_base['Date'], res_base['1'], res_recalib['1'], 'PCE')
plot(res_base['Date'], res_base['2'], res_recalib['2'], 'PBL')

times = pd.read_csv('times.csv')
times = times.rename(columns = {'0': 'Time'})
times1991 = pd.DataFrame(['-' for _ in range(7)], columns = ['Time'])
times1991.insert(0, 'Month', np.arange(1,8,1))
times1991.insert(0, 'Year', 1991)
times1991 = pd.concat([times1991, times[times['Year'] == 1991].round(2)], axis = 0, ignore_index=True)
times1992 = times[times['Year'] == 1992].round(2)
times1993 = times[times['Year'] == 1993].round(2)
times1994 = times[times['Year'] == 1994].round(2)
times1995 = times[times['Year'] == 1995].round(2)

times_concat = pd.concat([times1991, times1992, times1993, times1994, times1995], axis = 0).pivot(index = 'Month', columns = 'Year', values = 'Time').to_csv('times_grid.csv')
