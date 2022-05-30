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
from profile_generation import *
from plotting import *
import warnings
warnings.simplefilter('ignore', np.RankWarning)
from scipy.optimize import curve_fit
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from openpyxl import load_workbook
# from scipy.optimize import OptimizeWarning
# warnings.simplefilter("error", OptimizeWarning)

# +
# Generate profiles and join dataframes
patients_to_exclude_linear, patients_to_exclude_quad, list_of_patient_df, list_of_cal_pred_df, list_of_result_df = generate_profiles()
df, cal_pred, result_df = join_dataframes(list_of_patient_df, list_of_cal_pred_df, list_of_result_df)

# Print patients to exclude and ouput dataframes to excel as individual sheets
print_patients_to_exclude(patients_to_exclude_linear, patients_to_exclude_quad)
output_df_to_excel(df, cal_pred, result_df)

# +
# Plotting
# perc_days_within_target_tac(result_df)
# perc_days_outside_target_tac(result_df)
# median_perc_within_acc_dev(result_df)
# can_benefit(result_df)
# modified_TTR(result_df)
# wrong_range(result_df)

# +
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import math

d = {'dose': [0.5, 1, 1.5, 1.5, 3, 3], 'response': [2.4, 2.8, 3.2, 3.1, 7.9, 10]}
df = pd.DataFrame(data=d)

# Calculate weight
j = 0
decay_weight = []
for i in range(len(df)):
    decay_weight.append(math.exp(-(24*(i))/(12/np.log(2))))

# Fit model
poly_reg = PolynomialFeatures(degree=2)
X = np.array(df.dose).reshape(-1, 1)
y = np.array(df.response)
X = poly_reg.fit_transform(X)
result = LinearRegression(fit_intercept=False).fit(X, y, decay_weight)
new = np.array(3).reshape(-1, 1)
prediction = result.predict(poly_reg.fit_transform(new))[0]
prediction

# +
d = {'col1': [1, 2, 3], 'col2': [4, 5, 6], 'col3': np.nan}
df_1 = pd.DataFrame(data=d)

d = {'col1': [1, 2, 3], 'col2': [4, 5, 7], 'col3': np.nan}
df_2 = pd.DataFrame(data=d)

pd.concat([df_1, df_2])
# df['col3'] = ""
# df

