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
# ~5mins

# Execute CURATE without pop tau
execute_CURATE(pop_tau_string='')
# -


all_data()




def clinically_relevant_performance_metrics():
    """Clinically relevant performance metrics. 
    Calculate the results, conduct statistical tests, and
    print them out. 
    
    Instructions: Uncomment first block of code to write output to txt file.
    """
    # Uncomment to write output to txt file
    # file_path = 'Clinically relevant performance metrics.txt'
    # sys.stdout = open(file_path, "w")

    # Clinically relevant performance metrics. 
    # Calculate the results, conduct statistical tests, and
    # print them out. 

    # 1. Find percentage of days within clinically acceptable 
    # tac levels (6.5 to 12 ng/ml)

    data = pd.read_excel(all_data_file)

    # Add acceptable tacrolimus levels column
    data['acceptable'] = (data['response'] >= acceptable_tac_lower_limit) & (data['response'] <= acceptable_tac_upper_limit)

    # Calculate results
    acceptable_SOC = \
    data.groupby('patient')['acceptable'].apply(lambda x: x.sum()/x.count()*100).to_frame().reset_index()
    shapiro_p = stats.shapiro(acceptable_SOC.acceptable).pvalue
    if shapiro_p < 0.05:
        shapiro_result_string = 'reject normality'
    else:
        shapiro_result_string = 'assume normality'

    # Print results and normality test results
    print(f'(1) % of days within clinically acceptable tac levels:\n{acceptable_SOC.acceptable.describe()}\n')
    print(f'p-value = {shapiro_p:.2f}, Shapiro-Wilk test, {shapiro_result_string}')

    # 2. Find % of predictions within clinically acceptable
    # prediction error (between -1.5 and +2 ng/ml)

    dat = pd.read_excel(result_file, sheet_name='result')
    dat = dat[dat.method=='L_RW_wo_origin'].reset_index()
    dat = dat[['patient', 'pred_day', 'deviation']]

    # Add acceptable therapeutic range column
    dat['acceptable'] = (dat['deviation'] >=overprediction_limit) & (dat['deviation'] <= underprediction_limit)

    # Calculate results
    acceptable_CURATE = \
    dat.groupby('patient')['acceptable'].apply(lambda x: x.sum()/x.count()*100).to_frame().reset_index()
    shapiro_p = stats.shapiro(acceptable_CURATE.acceptable).pvalue
    if shapiro_p < 0.05:
        shapiro_result_string = 'reject normality'
    else:
        shapiro_result_string = 'assume normality'

    print(f'\n(2) % of predictions within clinically acceptable prediction error:\n{acceptable_CURATE.acceptable.describe()}\n')
    print(f'p-value = {shapiro_p:.2f}, Shapiro-Wilk test, {shapiro_result_string}\n')

    # Check for equal variance between (1) and (2): result p = 0.97, assume equal variance
    stat, p = levene(acceptable_SOC.acceptable, acceptable_CURATE.acceptable, center='mean')
    if p < 0.05:
        levene_result_string = 'unequal variance'
    else:
        levene_result_string = 'assume equal variance'
    print(f'p-value = {p:.2f}, Levene test, {levene_result_string}\n')

    # Check for equal means
    t_test = stats.ttest_ind(acceptable_SOC.acceptable, acceptable_CURATE.acceptable).pvalue
    if p < 0.05:
        result_string = 'unequal means'
    else:
        result_string = 'assume equal means'

    print(f'Comparison of means between (1) and (2), p-value = {p:.2f}, {result_string}\n')

    # 3. Clinically unacceptable overprediction

    # Add unacceptable overprediction
    dat['unacceptable_overprediction'] = (dat['deviation'] < overprediction_limit)
    dat['unacceptable_underprediction'] = (dat['deviation'] > underprediction_limit)

    unacceptable_overprediction = \
    dat.groupby('patient')['unacceptable_overprediction'].apply(lambda x: x.sum()/x.count()*100).to_frame().reset_index()
    result = stats.shapiro(unacceptable_overprediction.unacceptable_overprediction).pvalue
    if result < 0.05:
        result_string = 'reject normality'
    else:
        result_string = 'assume normality'

    print(f'(3) % clinically unacceptable overprediction:\n{unacceptable_overprediction.unacceptable_overprediction.describe()}\n')
    print(f'p-value = {result:.2f}, Shapiro-Wilk test, {result_string}\n')

    # 4. Clinically unacceptable underprediction

    # Add unacceptable underprediction
    unacceptable_underprediction = \
    dat.groupby('patient')['unacceptable_underprediction'].apply(lambda x: x.sum()/x.count()*100).to_frame().reset_index()
    result = stats.shapiro(unacceptable_underprediction.unacceptable_underprediction).pvalue
    if result < 0.05:
        result_string = 'reject normality'
    else:
        result_string = 'assume normality'

    print(f'(4) % clinically unacceptable underprediction:\n{unacceptable_underprediction.unacceptable_underprediction.describe()}\n')
    print(f'p-value = {result:.2f}, Shapiro-Wilk test, {result_string}\n')




dat = pd.read_excel('CURATE_results.xlsx', sheet_name='result')

df = dat.copy()
df.groupby('method')['deviation'].describe()

52.61-43.8



# +
input_file = 'Retrospective Liver Transplant Data - edited.xlsx'
rows_to_skip = 17
list_of_patient_df = []
patient = '84'

df = pd.read_excel(input_file, sheet_name=patient, skiprows=rows_to_skip)
df = clean_data(df)
# df = keep_ideal_data(df, patient, list_of_patient_df)
original_df = df.copy()
df_temp = df.copy()

# Create boolean column of data to remove
# Including NA, <2 tac level, multiple blood draws
df_temp['non_ideal'] = (df_temp.isnull().values.any(axis=1))  | \
                           (df_temp['Tac level (prior to am dose)'] == '<2') | \
                           (df_temp["Tac level (prior to am dose)"].astype(str).str.contains("/"))

# df_temp
# Set boolean for non_ideal as True if all dose including and above current row is 0
for i in range(len(df_temp)):
    if (df_temp.loc[0:i, 'Eff 24h Tac Dose'] == 0).all():
        df_temp.loc[i, 'non_ideal'] = True

# Create index column
df_temp.reset_index(inplace=True) 

# Find cumulative sum of data to be removed for each index row
df_cum_sum_non_ideal = df_temp['non_ideal'].cumsum()     

# # Find number of consecutive non-NA
# df_temp = df_temp.groupby(df_cum_sum_non_ideal).agg({'index': ['count', 'min', 'max']})

# # Groupby created useless column level for index, drop it
# df_temp.columns = df_temp.columns.droplevel()

# # Find largest chunk with consec non-NA
# df_temp = df_temp[df_temp['count']==df_temp['count'].max()] 
# df_temp.reset_index(inplace=True)

# # Find index of largest chunk to keep in dataframe
# if len(df_temp) > 1: # If there are >1 large chunks with longest length, an error will be printed
#     df_temp = print("there are >1 chunks of data with the longest length.")
# else:
#     # Find index of largest chunk to keep in dataframe
#     min_idx = df_temp.loc[0, 'min']
#     max_idx = df_temp.loc[0, 'max'] # Get max index where non-NA chunk ends

#     # Keep largest chunk in dataframe
#     df = df.iloc[min_idx:max_idx + 1, :] 

# # Format patient dataframe
# df['patient'] = patient
# df.columns = ['day', 'response', 'dose', 'patient']

print(f'df_cum_sum_non_ideal\n {df_cum_sum_non_ideal} |\ndf_temp {df_temp}')
# -

all_data()

# +
df = pd.read_excel('Retrospective Liver Transplant Data - edited.xlsx', sheet_name='84', skiprows=17)

# Keep target columns
df = df[["Day #", "Tac level (prior to am dose)", "Eff 24h Tac Dose"]]

# Shift tac level one cell up to match dose-response to one day
df['Eff 24h Tac Dose'] = df['Eff 24h Tac Dose'].shift(1)

# Remove "mg"/"ng" from dose
df['Eff 24h Tac Dose'] = df['Eff 24h Tac Dose'].astype(str).str.replace('mg', '')
df['Eff 24h Tac Dose'] = df['Eff 24h Tac Dose'].astype(str).str.replace('ng', '')
df['Eff 24h Tac Dose'] = df['Eff 24h Tac Dose'].astype(float)

first_day_of_dosing = df['Day #'].loc[~df['Eff 24h Tac Dose'].isnull()].iloc[0]

# Keep data from first day of dosing
df = df[df['Day #'] >= first_day_of_dosing].reset_index(drop=True)

# Set the first day of dosing as day 2 (because first dose received on day 1)
for i in range(len(df)):
    df.loc[i, 'Day #'] = i + 2

original_df = df.copy()
df_temp = df.copy()

# Create boolean column of data to remove
# Including NA, <2 tac level, multiple blood draws
df_temp['non_ideal'] = (df_temp.isnull().values.any(axis=1))  | \
                           (df_temp['Tac level (prior to am dose)'] == '<2') | \
                           (df_temp["Tac level (prior to am dose)"].astype(str).str.contains("/"))

# Set boolean for non_ideal as True if all dose including and above current row is 0
for i in range(len(df_temp)):
    if (df_temp.loc[0:i, 'Eff 24h Tac Dose'] == 0).all():
        df_temp.loc[i, 'non_ideal'] = True

# Create index column
df_temp.reset_index(inplace=True) 

# Find cumulative sum of data to be removed for each index row
df_cum_sum_non_ideal = df_temp['non_ideal'].cumsum()     

# Find number of consecutive non-NA
df_temp = df_temp.groupby(df_cum_sum_non_ideal).agg({'index': ['count', 'min', 'max']})

# Groupby created useless column level for index, drop it
df_temp.columns = df_temp.columns.droplevel()

# Find largest chunk with consec non-NA
df_temp = df_temp[df_temp['count']==df_temp['count'].max()] 
df_temp.reset_index(inplace=True)

# Find index of largest chunk to keep in dataframe
if len(df_temp) > 1: # If there are >1 large chunks with longest length, an error will be printed
    df_temp = print("there are >1 chunks of data with the longest length.")
else:
    # Find index of largest chunk to keep in dataframe
    min_idx = df_temp.loc[0, 'min'] 
    max_idx = df_temp.loc[0, 'max'] # Get max index where non-NA chunk ends

    # Keep largest chunk in dataframe
    df = df.iloc[min_idx:max_idx + 1, :] 

# Format patient dataframe
df['patient'] = patient
df.columns = ['day', 'response', 'dose', 'patient']

print(f'df_cum_sum_non_ideal\n {df_cum_sum_non_ideal} \n df\n {df_temp} \n min_idx: {min_idx} | max_idx {max_idx} \n df\n {df} \n original_df \n {original_df}')

# +
patient='84'

patient_df = pd.read_excel('Retrospective Liver Transplant Data - edited.xlsx', sheet_name=patient, skiprows=17)
patient_df['patient'] = patient

# Subset dataframe
patient_df = patient_df[['Day #', 'Tac level (prior to am dose)', 'Eff 24h Tac Dose', 'patient']]

# Shift dose column
patient_df['Eff 24h Tac Dose'] = patient_df['Eff 24h Tac Dose'].shift(1)

# Remove "mg"/"ng" from dose
patient_df['Eff 24h Tac Dose'] = patient_df['Eff 24h Tac Dose'].astype(str).str.replace('mg', '')
patient_df['Eff 24h Tac Dose'] = patient_df['Eff 24h Tac Dose'].astype(str).str.replace('ng', '')
patient_df['Eff 24h Tac Dose'] = patient_df['Eff 24h Tac Dose'].astype(float)

first_day_of_dosing = patient_df['Day #'].loc[~patient_df['Eff 24h Tac Dose'].isnull()].iloc[0]

# Keep data from first day of dosing
patient_df = patient_df[patient_df['Day #'] >= first_day_of_dosing].reset_index(drop=True)

# Set the first day of dosing as day 2 (because first dose received on day 1)
for i in range(len(patient_df)):
    patient_df.loc[i, 'Day #'] = i + 2
    
patient_df

# +
# Find % of days within clinically acceptable tac levels

df = read_excel('all_data.xlsx')
df
# -



# +
# %%time
# Perform CV
five_fold_cross_val_results, five_fold_cross_val_results_summary = find_pop_tau_with_CV()
execute_CURATE_and_update_pop_tau_results('CV', five_fold_cross_val_results_summary, five_fold_cross_val_results)

# Perform LOOCV
five_fold_cross_val_results, five_fold_cross_val_results_summary = find_pop_tau_with_LOOCV()
execute_CURATE_and_update_pop_tau_results('LOOCV', five_fold_cross_val_results_summary, five_fold_cross_val_results)
