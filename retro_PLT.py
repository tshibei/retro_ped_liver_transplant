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

whole_day_original = pd.read_excel('CURATE_results.xlsx', sheet_name='result')
evening_original = pd.read_excel('CURATE_results_evening_dose.xlsx', sheet_name='result')

# +
whole_day = whole_day_original.copy()
evening = evening_original.copy()

# Check if equal means in deviation: result, p = 0.75, equal
whole_day_deviation = whole_day[whole_day.method=="L_RW_wo_origin"].deviation.reset_index(name='deviation')[['deviation']]
evening_deviation = evening[evening.method=="L_RW_wo_origin"].deviation.reset_index(name='deviation')[['deviation']]

print(f'deviation p-value: {stats.ttest_rel(whole_day_deviation.deviation, evening_deviation.deviation).pvalue:.2f}')
print(f"deviation mean, whole_day = {whole_day_deviation.describe().loc['mean'][0]:.2f}")
print(f"deviation mean, evening = {evening_deviation.describe().loc['mean'][0]:.2f}\n")

# Check if equal means in abs_deviation: result p = 0.86, equal
whole_day_abs_deviation = whole_day[whole_day.method=="L_RW_wo_origin"].abs_deviation.reset_index(name='abs_deviation')[['abs_deviation']]
evening_abs_deviation = evening[evening.method=="L_RW_wo_origin"].abs_deviation.reset_index(name='abs_deviation')[['abs_deviation']]

print(f'abs deviation p-value: {stats.ttest_rel(whole_day_abs_deviation.abs_deviation, evening_abs_deviation.abs_deviation).pvalue:.2f}')
print(f"abs deviation mean, whole_day = {whole_day_abs_deviation.describe().loc['mean'][0]:.2f}")
print(f"abs deviation mean, evening = {evening_abs_deviation.describe().loc['mean'][0]:.2f}")
# print(f'whole_day:\n{whole_day[whole_day.method=="L_RW_wo_origin"].deviation.describe()}')
# print(f'evening:\n{evening[evening.method=="L_RW_wo_origin"].deviation.describe()}')
# -



# +
# %%time
# ~5mins

execute_CURATE()

# +
# %%time
# Perform CV
five_fold_cross_val_results, five_fold_cross_val_results_summary = find_pop_tau_with_CV()
execute_CURATE_and_update_pop_tau_results('CV', five_fold_cross_val_results_summary, five_fold_cross_val_results)

# Perform LOOCV
five_fold_cross_val_results, five_fold_cross_val_results_summary = find_pop_tau_with_LOOCV()
execute_CURATE_and_update_pop_tau_results('LOOCV', five_fold_cross_val_results_summary, five_fold_cross_val_results)
