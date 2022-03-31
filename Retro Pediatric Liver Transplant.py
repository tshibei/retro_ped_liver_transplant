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

# Create list of patient numbers
patient_list = ['84', '114', '117', '118', '120', '121', '122', '123', '125', '126', 
               '129', '130', '131', '132', '133', '138']

# Create dictionary of dataframes for each patient
df = {}

# Set parameters
rows_to_skip = 17 # Number of rows to skip before reaching patient tac data

# Read each patient's data into a dataframe
for patient in patient_list:
    
    # Read individual patient data from excel, shift tac level one cell up, remove "mg" from values
    df[patient] = read_indiv_patient_data(input_file, patient, rows_to_skip)

    # Keep largest consecutive non-NA chunk of patient data
    df[patient] = keep_longest_chunk(df[patient]) # If there are >1 large chunks with longest length, an error will be printed
    
    df[patient] = df[patient].reset_index(drop=True) 

    print(patient_name, df[patient])
    
# +
# # Perform normality test, both numerical and graphical
# normality_test(df)

# Generate predictions and calculate deviations using different methods
# df_Q_Cum = Q_Cum(df)
# df_Q_PPM = Q_PPM(df)
# df_Q_RW = Q_RW(df)
# df_L_Cum = L_Cum(df)
# df_L_PPM = L_PPM(df)
# df_L_RW = L_RW(df)
# df_Q_Cum_0 = Q_Cum_0(df)
# df_Q_PPM_0 = Q_PPM_0(df)
# df_Q_RW_0 = Q_RW_0(df)
# df_L_Cum_0 = L_Cum_0(df)
# df_L_PPM_0 = L_PPM_0(df)
# df_L_RW_0 = L_RW_0(df)

# # Create dataframe of deviation
# data_frames = [df_Q_Cum[['prediction day', 'deviation']],
#              df_Q_PPM[['prediction day', 'deviation']],
#              df_Q_RW[['prediction day', 'deviation']],
#              df_L_Cum[['prediction day', 'deviation']],
#              df_L_PPM[['prediction day', 'deviation']],
#              df_L_RW[['prediction day', 'deviation']]]
# df_deviation = reduce(lambda  left,right: pd.merge(left,right,on=['prediction day'],
#                                             how='outer'), data_frames)

# # Keep rows with common prediction day
# df_deviation = df_deviation.iloc[:-1,:]
# df_deviation.columns = ['pred_day', 'Q_Cum', 'Q_PPM', 'Q_RW', 'L_Cum', 'L_PPM', 'L_RW']

# +
patient_name = '125'
rows_to_skip = 17 # Number of rows to skip before reaching patient tac data

# Read individual patient data from excel, shift tac level one cell up, remove "mg" from values
df = read_indiv_patient_data(input_file, patient_name, rows_to_skip)

# Keep largest consecutive non-NA chunk of patient data
df = keep_longest_chunk(df) # If there are >1 large chunks with longest length, an error will be printed

df = df.reset_index(drop=True)
df = df.reset_index(drop=False)

df['']
# Out of the first 3 unique tac doses, create list of indexes of the latest appearances
# unique_doses = df.sort_values('Day #').groupby(['Eff 24h Tac Dose']).tail(1)
# .reset_index()['index'].tolist()
df.loc[df.groupby('Eff 24h Tac Dose').['Eff 24h Tac Dose'].idxmax()]
# # Create dataframe of latest appearances of the first 3 unique tac doses
# unique_df = df.iloc[[unique_doses[0], unique_doses[1], unique_doses[2]],:]
# unique_df = unique_df.drop(labels='index', axis=1) # Drop useless index column

# # Join first 3 unique tac doses with the rest of data in selected chunk
# df = pd.concat([unique_df, df.iloc[unique_doses[2]+1:,:]]) 

# df = df.drop(labels='index', axis=1)
# df = df.reset_index()

# df
# # Perform normality test, both numerical and graphical
# normality_test(df)

# print(patient_name, df)

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
