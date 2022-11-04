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
from plotting import *
import statistics
import seaborn as sns
from scipy.stats import wilcoxon
import pandas as pd


dosing_strategy_values()

case_series_118_repeated_dosing_response_vs_dose(plot=True)
