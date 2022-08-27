# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.13.7
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %load_ext autoreload
# %autoreload 2
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from scipy import stats
import seaborn as sns
from functools import reduce
pd.options.mode.chained_assignment = None 
from statistics import mean
from Profile_Generation import *
from plotting import *
import warnings
warnings.simplefilter('ignore', np.RankWarning)
from scipy.optimize import curve_fit
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from openpyxl import load_workbook
import math
from scipy.optimize import OptimizeWarning
warnings.simplefilter("ignore", OptimizeWarning)
import timeit
from sklearn.metrics import mean_squared_error
from matplotlib.pyplot import cm
from matplotlib.patches import Patch
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import levene
import sys

# +
# %%time
# Retrieve dose, response, day, for patient 120
result_file = 'CURATE_results.xlsx'
dat = pd.read_excel(result_file, sheet_name='result')

dat
# -

df = dat.copy()
df

# +
dat_original, combined_df = case_series_120()

# Subset first prediction since it could outperform SOC
dat = combined_df[combined_df.pred_day==4].reset_index(drop=True)

sns.set(style='white', font_scale=2,
       rc={"figure.figsize":(7,7), "xtick.bottom":True, "ytick.left":True})

# Plot regression line
x = np.array([dat.x[0],dat.x[1]])
y = np.array([dat.y[0],dat.y[1]])
a, b = np.polyfit(x, y, 1)
x_values = np.linspace(0, 3)
plt.plot(x_values, a*x_values + b, linestyle='-', color='y')

# Plot scatter points
plt.scatter(x, y, s=100, color='y')

# Plot therapeutic range
plt.axhspan(8, 10, facecolor='grey', alpha=0.2)

# Label days
for i in range(dat.shape[0]):
    plt.text(x=dat.x[i]+0.1,y=dat.y[i]+0.1,s=int(dat.day[i]),
             fontdict=dict(color='black',size=13),
             bbox=dict(facecolor='y', ec='black', alpha=0.5, boxstyle='circle'))

sns.despine()
plt.title('Day 4 recommendation')
plt.xlabel('Dose (mg)')
plt.ylabel('Tacrolimus level (ng/ml)')
plt.xticks(np.arange(0,3.5,step=0.5))
plt.xlim(0,2.5)

# +
# %%time
# ~5mins

# Execute CURATE without pop tau
execute_CURATE(pop_tau_string='')


# +
# %%time
# Perform CV
five_fold_cross_val_results, five_fold_cross_val_results_summary = find_pop_tau_with_CV()
execute_CURATE_and_update_pop_tau_results('CV', five_fold_cross_val_results_summary, five_fold_cross_val_results)

# Perform LOOCV
five_fold_cross_val_results, five_fold_cross_val_results_summary = find_pop_tau_with_LOOCV()
execute_CURATE_and_update_pop_tau_results('LOOCV', five_fold_cross_val_results_summary, five_fold_cross_val_results)
