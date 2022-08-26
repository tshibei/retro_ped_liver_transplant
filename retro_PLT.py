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

clinically_relevant_performance_metrics()

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
# -

data = response_vs_day(plot=False)
data.dose.describe()

# +
# Compare between dropping NaN vs no dropping
data = response_vs_day(plot=False)

# Drop rows where response is NaN
data = data[data.response.notna()].reset_index(drop=True)

# Add therapeutic range column
for i in range(len(data)):
    if (data.response[i] >= therapeutic_range_lower_limit) & (data.response[i] <= therapeutic_range_upper_limit):
        data.loc[i, 'therapeutic_range'] = True
    else:
        data.loc[i, 'therapeutic_range'] = False

perc_therapeutic_range = data.groupby('patient')['therapeutic_range'].apply(lambda x: x.sum()/x.count()*100)
perc_therapeutic_range = perc_therapeutic_range.to_frame().reset_index()

# Result and distribution
result_and_distribution(perc_therapeutic_range.therapeutic_range, 'drop NaN')



# +
# Compare between dropping NaN vs no dropping
data = response_vs_day(plot=False)

# Drop rows where response is NaN
data = data[data.response.notna()].reset_index(drop=True)

# Add therapeutic range column
for i in range(len(data)):
    if (data.response[i] >= therapeutic_range_lower_limit) & (data.response[i] <= therapeutic_range_upper_limit):
        data.loc[i, 'therapeutic_range'] = True
    else:
        data.loc[i, 'therapeutic_range'] = False

perc_therapeutic_range = data.groupby('patient')['therapeutic_range'].apply(lambda x: x.sum()/x.count()*100)
perc_therapeutic_range = perc_therapeutic_range.to_frame().reset_index()

# Result and distribution
result_and_distribution(perc_therapeutic_range.therapeutic_range, 'keep NaN')
