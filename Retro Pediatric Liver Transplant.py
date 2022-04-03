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
import warnings
warnings.simplefilter('ignore', np.RankWarning)
from scipy.optimize import curve_fit

# +
from scipy.optimize import curve_fit
input_file = 'Retrospective Liver Transplant Data.xlsx'

# Create dictionaries to store information for individual patients
df = {}
cal_pred = {}
df_Q_Cum = {}
df_Q_Cum_origin_int = {}
df_Q_PPM = {}
df_Q_PPM_origin_int = {}
df_L_Cum = {}
df_L_Cum_origin_int = {}
df_L_PPM = {}
df_L_PPM_origin_int = {}
df_Q_RW_input = {}
df_Q_RW = {}
df_L_RW_input = {}
df_L_RW = {}
df_RW = {}

# Define lists and parameters
patients_to_exclude = []
rows_to_skip = 17 # Number of rows to skip before reaching patient tac data
patient_list = ['84', '114', '117', '118', '120', '121', '122', '123', '125', '126', 
               '129', '130', '131', '132', '133', '138']

# Loop through patients
for patient in patient_list:
    
    # 1. Data cleaning: 
    
    # Read individual patient data from excel, shift tac level one cell up, remove "mg" and "ng" from values
    df[patient] = read_indiv_patient_data(input_file, patient, rows_to_skip)

    # 2. Data selection: 
    
    # Keep ideal data only
    df[patient] = keep_ideal_data(df[patient]) # If there are >1 large chunks with longest length, an error will be printed
    df[patient] = df[patient].reset_index(drop=True) 
    
    # Select data for calibration and subsequent predictions
    # Print patients with insufficient data for calibration and with <3 predictions
    cal_pred[patient] = select_calibration_prediction_data(df, patient, cal_pred,
                                                          patients_to_exclude)
    
# Print list of patients to exclude generated from cal_pred function
print("Patients to exclude from CURATE.AI predictions: ", patients_to_exclude)

# Exclude chosen patients from list
patient_list = [patient for patient in patient_list if patient not in patients_to_exclude]

# 3. Apply CURATE.AI methods to all remaining patients:

# Loop through patients
for patient in patient_list:

    # Perform all methods except rolling window
    df_Q_Cum[patient] = Q_Cum(cal_pred[patient])
    df_Q_Cum_origin_int[patient] = Q_Cum_origin_int(cal_pred[patient])
    df_Q_PPM[patient] = Q_PPM(cal_pred[patient])
    df_Q_PPM_origin_int[patient] = Q_PPM_origin_int(cal_pred[patient])
    df_L_Cum[patient] = L_Cum(cal_pred[patient])
    df_L_Cum_origin_int[patient] = L_Cum_origin_int(cal_pred[patient])
    df_L_PPM[patient] = L_PPM(cal_pred[patient])
    df_L_PPM_origin_int[patient] = L_PPM_origin_int(cal_pred[patient])
    
    # Perform rolling window methods
    df_Q_RW_input[patient] = select_RW_data(cal_pred[patient], 3) # Extra data selection step for RW
    df_Q_RW[patient] = RW(df_Q_RW_input[patient], patient, df_RW, 3)
    df_L_RW_input[patient] = select_RW_data(cal_pred[patient], 2) # Extra data selection step for RW
    df_L_RW[patient] = RW(df_L_RW_input[patient], patient, df_RW, 2)
    
# 4. Plot results
cal_pred_dataframe = cal_pred['114']

# -


prediction_dataframe = {}
index_first_prediction = 4 # linear is 3, quadratic is 4
deg = 2 # degree of fitting polynomial: linear is 1, quadratic is 2
patient = '114'
prediction_dataframe[patient] = prep_Cum_origin_dp_data(cal_pred[patient], index_first_prediction, deg)
df_Cum_origin_dp = {}
df_Cum_origin_dp[patient] = Cum_origin_dp(df_Cum_origin_dp, patient, prediction_dataframe[patient], 4, 2)
df_Cum_origin_dp[patient]

cal_pred_dataframe = cal_pred['114'].copy()
len(cal_pred_dataframe)

prediction_dataframe.loc[i, 'Dose_1':'Dose_' + str(max_cum_length)]



# +
methods = ['Q_Cum', 'Q_PPM', 'Q_RW', 'L_Cum', 'L_PPM', 'L_RW']

df_Q_Cum['method'] = 'Q_Cum'
df_Q_PPM['method'] = 'Q_PPM'
df_Q_RW['method'] = 'Q_RW'
df_L_Cum['method'] = 'L_Cum'
df_L_PPM['method'] = 'L_PPM'
df_L_RW['method'] = 'L_RW'
df_Q_Cum_0['method'] = 'Q_Cum_0'
df_Q_PPM_0['method'] = 'Q_PPM_0'
df_Q_RW_0['method'] = 'Q_RW_0'
df_L_Cum_0['method'] = 'L_Cum_0'
df_L_PPM_0['method'] = 'L_PPM_0'
df_L_RW_0['method'] = 'L_RW_0'
df_all_methods = pd.concat([df_Q_Cum, df_Q_PPM, df_Q_RW, df_L_Cum, df_L_PPM, df_L_RW,
                           df_Q_Cum_0, df_Q_PPM_0, df_Q_RW_0, df_L_Cum_0, df_L_PPM_0, df_L_RW_0])
df_all_methods = df_all_methods.reset_index(drop = True)

# Remove rows with prediction day 4
df_all_methods.drop(df_all_methods[df_all_methods['prediction day'] <= 4].index, inplace=True)
df_all_methods = df_all_methods.reset_index(drop = True)
df_all_methods.columns = ['pred_day', 'a', 'b', 'c',
                                       'prediction', 'deviation', 'abs_dev', 'method']


df_all_methods['method'] = df_all_methods['method'].astype('category')
df_all_methods['method'].cat.reorder_categories(['Q_Cum', 'Q_PPM', 'Q_RW', 
                                                 'L_Cum', 'L_PPM', 'L_RW',
                                                'Q_Cum_0', 'Q_PPM_0', 'Q_RW_0', 
                                                 'L_Cum_0', 'L_PPM_0', 'L_RW_0'])

# create color mapping based on all unique values of ticker
method = df_all_methods.method.unique()
colors = sns.color_palette('Paired')  # get a number of colors
cmap = dict(zip(method, colors))  # zip values to colors

# plot
plt.figure(figsize=(16, 10))
sns.lineplot(x='pred_day', y='deviation', hue='method', data=df_all_methods, palette=cmap)

# plt.tight_layout()
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.xlabel("Day of Prediction")
plt.ylabel("Deviation")
plt.title("Deviation of Prediction from Actual Value")

# plt.savefig('test.png', bbox_inches="tight", dpi=300)

# +
# # Plot mean deviation

methods = ['L_Cum', 'L_Cum_0', 'L_PPM', 'L_PPM_0', 'L_RW', 'L_RW_0', 
           'Q_Cum', 'Q_Cum_0', 'Q_PPM', 'Q_PPM_0', 'Q_RW', 'Q_RW_0']
x_pos = np.arange(len(methods))
CTEs = df_all_methods.groupby("method").deviation.mean()
error = df_all_methods.groupby("method").deviation.std()

# Build the plot
fig, ax = plt.subplots()
ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
ax.set_ylabel('Deviation (Mean \u00B1 SD)')
ax.set_xticks(x_pos)
ax.set_xticklabels(methods)
ax.set_title('Deviation of Predicted from Actual Value (Mean \u00B1 SD)')
ax.yaxis.grid(True)
plt.xticks(rotation=45)

# Save the figure and show
plt.tight_layout()
# plt.savefig('all_methods_mean_deviation.png', bbox_inches="tight", dpi=300)
plt.show()

error

# +
# Plot median of deviation
methods = ['L_Cum', 'L_Cum_0', 'L_PPM', 'L_PPM_0', 'L_RW', 'L_RW_0', 
           'Q_Cum', 'Q_Cum_0', 'Q_PPM', 'Q_PPM_0', 'Q_RW', 'Q_RW_0']
x_pos = np.arange(len(methods))

df = df_all_methods.pivot("pred_day", "method" , "deviation")

data = [df['L_Cum'], df['L_Cum_0'], df['L_PPM'], df['L_PPM_0'], df['L_RW'], df['L_RW_0'], 
           df['Q_Cum'], df['Q_Cum_0'], df['Q_PPM'], df['Q_PPM_0'], df['Q_RW'], df['Q_RW_0']]

fig, ax = plt.subplots()
ax.set_title('Deviation of Predicted from Actual Value (Median)')
ax.boxplot(data)
ax.set_xticklabels(methods)
plt.ylabel('Deviation (Median)')
plt.xticks(rotation=45)
# plt.savefig('all_methods_median_dev.png', bbox_inches="tight", dpi=300)
plt.show()

# +
# Plot RMSE and MAE
df_rmse_MAE = pd.DataFrame()

## Plot RMSE
methods = ['L_Cum', 'L_Cum_0', 'L_PPM', 'L_PPM_0', 'L_RW', 'L_RW_0', 
           'Q_Cum', 'Q_Cum_0', 'Q_PPM', 'Q_PPM_0', 'Q_RW', 'Q_RW_0']

df = df_all_methods.pivot("pred_day", "method" , "deviation")

rmse_Q_Cum = np.sqrt(mean(df['Q_Cum']**2))
rmse_Q_PPM = np.sqrt(mean(df['Q_PPM']**2))
rmse_Q_RW = np.sqrt(mean(df['Q_RW']**2))
rmse_L_Cum = np.sqrt(mean(df['L_Cum']**2))
rmse_L_PPM = np.sqrt(mean(df['Q_Cum']**2))
rmse_L_RW = np.sqrt(mean(df['L_RW']**2))
rmse_Q_Cum_0 = np.sqrt(mean(df['Q_Cum_0']**2))
rmse_Q_PPM_0 = np.sqrt(mean(df['Q_PPM_0']**2))
rmse_Q_RW_0 = np.sqrt(mean(df['Q_RW_0']**2))
rmse_L_Cum_0 = np.sqrt(mean(df['L_Cum_0']**2))
rmse_L_PPM_0 = np.sqrt(mean(df['Q_Cum_0']**2))
rmse_L_RW_0 = np.sqrt(mean(df['L_RW_0']**2))

rmse = np.array([rmse_L_Cum, rmse_L_Cum_0, rmse_L_PPM, rmse_L_PPM_0, rmse_L_RW, rmse_L_RW_0,
                rmse_Q_Cum, rmse_Q_Cum_0, rmse_Q_PPM, rmse_Q_PPM_0, rmse_Q_RW, rmse_Q_RW_0])

rmse = pd.DataFrame(rmse.reshape(-1, len(rmse)),columns=methods)
rmse=rmse.transpose()

## Calculate MAE
MAE_Q_Cum = mean(abs(df['Q_Cum']))
MAE_Q_PPM = mean(abs(df['Q_PPM']))
MAE_Q_RW = mean(abs(df['Q_RW']))
MAE_L_Cum = mean(abs(df['L_Cum']))
MAE_L_PPM = mean(abs(df['L_PPM']))
MAE_L_RW = mean(abs(df['L_RW']))
MAE_Q_Cum_0 = mean(abs(df['Q_Cum_0']))
MAE_Q_PPM_0 = mean(abs(df['Q_PPM_0']))
MAE_Q_RW_0 = mean(abs(df['Q_RW_0']))
MAE_L_Cum_0 = mean(abs(df['L_Cum_0']))
MAE_L_PPM_0 = mean(abs(df['L_PPM_0']))
MAE_L_RW_0 = mean(abs(df['L_RW_0']))

MAE = np.array([MAE_L_Cum, MAE_L_Cum_0, MAE_L_PPM, MAE_L_PPM_0, MAE_L_RW, MAE_L_RW_0,
                MAE_Q_Cum, MAE_Q_Cum_0, MAE_Q_PPM, MAE_Q_PPM_0, MAE_Q_RW, MAE_Q_RW_0])

MAE = pd.DataFrame(MAE.reshape(-1, len(MAE)),columns=methods)
MAE=MAE.transpose()

df_rmse_MAE = df_rmse_MAE.append(rmse)
df_rmse_MAE = pd.concat([df_rmse_MAE, MAE], axis=1)
df_rmse_MAE.columns = ['RMSE', 'MAE']

df_rmse_MAE.index=['L_Cum', 'L_Cum_0', 'L_PPM', 'L_PPM_0', 'L_RW', 'L_RW_0', 
           'Q_Cum', 'Q_Cum_0', 'Q_PPM', 'Q_PPM_0', 'Q_RW', 'Q_RW_0']

df_rmse_MAE.plot()
plt.xticks(np.arange(len(df_rmse_MAE.index)), df_rmse_MAE.index, rotation=45)

plt.ylabel('RMSE and MAE')
plt.title("RMSE and MAE of Deviation of Predicted from Actual Value")
plt.xticks(rotation=45)
# plt.savefig('all_methods_MAE_RMSE.png', bbox_inches="tight", dpi=300)
# -

# Plot prediction of all methods
pred = df_all_methods.pivot("pred_day", "method", "prediction")
pred['L_Cum'].plot()
pred['L_Cum_0'].plot()
pred['L_PPM'].plot()
pred['L_PPM_0'].plot()
pred['L_RW'].plot()
pred['L_RW_0'].plot()
pred['Q_Cum'].plot()
pred['Q_Cum_0'].plot()
pred['Q_PPM'].plot()
pred['Q_PPM_0'].plot()
pred['Q_RW'].plot()
pred['Q_RW_0'].plot()
plt.axhline(y = 8, color = "black", linestyle = '--')
plt.axhline(y = 10, color = "black", linestyle = '--')
plt.legend()
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.xlabel("Day of Prediction")
plt.ylabel("Prediction")
plt.title("Prediction of Tac Level")
# plt.title("Prediction of Tac Level (without Q_PPM, Q_PPM_0, Q_RW, Q_RW_0)")
plt.tight_layout()
# plt.savefig('prediction.png', bbox_inches="tight", dpi=300)
