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
import matplotlib.patches as patches

sections = ['main', 'errdict', 'excdict']
section_dict = {sec : {} for sec in sections}
section_dict['main']


# +
from scipy.optimize import curve_fit
input_file = 'Retrospective Liver Transplant Data.xlsx'

# Create dictionaries to store information for individual patients
# df = {}
# quad_cal_pred = {}
# linear_cal_pred = {}
# df_Q_Cum = {}
# df_Q_Cum_input = {}
# df_Q_Cum_origin_int_input = {}
# df_Q_Cum_origin_int = {}
# df_Q_PPM = {}
# df_Q_PPM_origin_int = {}
# df_L_Cum = {}
# df_L_Cum_input = {}
# df_L_Cum_origin_int = {}
# df_L_Cum_origin_int_input = {}
# df_L_PPM = {}
# df_L_PPM_origin_int = {}
# df_Q_RW_input = {}
# df_Q_RW = {}
# df_L_RW_input = {}
# df_L_RW = {}
# df_RW = {}
# df_Q_Cum_origin_dp_input = {}
# df_Q_Cum_origin_dp = {}
# df_L_Cum_origin_dp_input = {}
# df_L_Cum_origin_dp = {}
# df_Q_PPM_origin_dp = {}
# df_L_PPM_origin_dp = {}
# df_PPM_origin_dp = {}

# dicts = ['df', 'quad_cal_pred', 'linear_cal_pred', 
#         'df_Q_Cum', 'df_Q_Cum_input', 'df_Q_Cum_origin_int_input', 'df_Q_Cum_origin_int',
#         'df_Q_PPM', 'df_Q_PPM_origin_int',
#         'df_L_Cum', 'df_L_Cum_input', 'df_L_Cum_origin_int', 'df_L_Cum_origin_int_input',
#         'df_L_PPM', 'df_L_PPM_origin_int',
#         'df_Q_RW_input', 'df_Q_RW', 'df_L_RW_input', 'df_L_RW', 'df_RW',
#         'df_Q_Cum_origin_dp_input', 'df_Q_Cum_origin_dp',
#         'df_L_Cum_origin_dp_input', 'df_L_Cum_origin_dp',
#         'df_Q_PPM_origin_dp', 'df_L_PPM_origin_dp', 'df_PPM_origin_dp']

# # for y in enumerate(dicts):
# #     y = {}

# all_dicts = {each_dict : {} for each_dict in dicts}

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
    quad_cal_pred[patient] = cal_pred_data(df, patient, quad_cal_pred, patients_to_exclude, 2)
    linear_cal_pred[patient] = cal_pred_data(df, patient, linear_cal_pred, patients_to_exclude, 1)

# Print list of unique patients to exclude generated from cal_pred function
patients_to_exclude = np.array(patients_to_exclude)
patients_to_exclude = np.unique(patients_to_exclude)
print("Patients to exclude from CURATE.AI predictions: ", patients_to_exclude)

# Exclude chosen patients from list
patient_list = [patient for patient in patient_list if patient not in patients_to_exclude]

# 3. Apply CURATE.AI methods to all remaining patients:

# Loop through patients
for patient in patient_list:

    # Perform all methods except rolling window and origin_dp methods
    df_Q_Cum_input[patient], df_Q_Cum[patient] = Q_Cum(quad_cal_pred[patient])
    df_Q_Cum_origin_int_input[patient], df_Q_Cum_origin_int[patient] = Q_Cum_origin_int(quad_cal_pred[patient])
    df_Q_PPM[patient] = Q_PPM(quad_cal_pred[patient])
    df_Q_PPM_origin_int[patient] = Q_PPM_origin_int(quad_cal_pred[patient])
    df_L_Cum_input[patient], df_L_Cum[patient] = L_Cum(linear_cal_pred[patient])
    df_L_Cum_origin_int_input[patient], df_L_Cum_origin_int[patient] = L_Cum_origin_int(linear_cal_pred[patient])
    df_L_PPM[patient] = L_PPM(linear_cal_pred[patient])
    df_L_PPM_origin_int[patient] = L_PPM_origin_int(linear_cal_pred[patient])
    
    # Perform rolling window methods which require extra data selection step
    df_Q_RW_input[patient] = select_RW_data(quad_cal_pred[patient], 3) 
    df_Q_RW[patient] = RW(df_Q_RW_input[patient], patient, df_RW, 3)
    df_L_RW_input[patient] = select_RW_data(linear_cal_pred[patient], 2)
    df_L_RW[patient] = RW(df_L_RW_input[patient], patient, df_RW, 2)
    
    # Perform Cumulative origin_dp methods with require extra data selection step
    df_Q_Cum_origin_dp_input[patient] = prep_Cum_origin_dp_data(quad_cal_pred[patient], 4, 2)
    df_Q_Cum_origin_dp[patient] = Cum_origin_dp(df_Q_Cum_origin_dp, patient, df_Q_Cum_origin_dp_input[patient], 4, 2)
    df_L_Cum_origin_dp_input[patient] = prep_Cum_origin_dp_data(linear_cal_pred[patient], 3, 1)
    df_L_Cum_origin_dp[patient] = Cum_origin_dp(df_L_Cum_origin_dp, patient, df_L_Cum_origin_dp_input[patient], 3, 1)
    
    # Perform PPM origin_dp methods
    df_Q_PPM_origin_dp[patient] = PPM_origin_dp(quad_cal_pred[patient], 2, df_Q_PPM_origin_dp, patient)
    df_L_PPM_origin_dp[patient] = PPM_origin_dp(linear_cal_pred[patient], 1, df_L_PPM_origin_dp, patient)

# 4. Export results for checking (will refactor this giant part later!)

# Create list of method dictionaries
dict_list = [df_Q_Cum, df_Q_Cum_origin_int, df_Q_Cum_origin_dp,
            df_Q_PPM, df_Q_PPM_origin_int, df_Q_PPM_origin_dp, df_Q_RW,
            df_L_Cum, df_L_Cum_origin_int, df_L_Cum_origin_dp,
            df_L_PPM, df_L_PPM_origin_int, df_L_PPM_origin_dp,
            df_L_RW]

# Create list of method names
method_names = ["Q_Cum", "Q_Cum_origin_int", "Q_Cum_origin_dp",
                "Q_PPM", "Q_PPM_origin_int", "Q_PPM_origin_dp", "Q_RW",
                "L_Cum", "L_Cum_origin_int", "L_Cum_origin_dp",
                "L_PPM", "L_PPM_origin_int", "L_PPM_origin_dp", "L_RW"]

# Create list of method_input dictionaries
input_dict_list = [df_Q_Cum_input, df_Q_Cum_origin_int_input, df_Q_Cum_origin_dp_input, df_Q_RW_input,
                   df_L_Cum_input, df_L_Cum_origin_int_input, df_L_Cum_origin_dp_input, df_L_RW_input]

# Create list of method names for prediction input dataframes
input_method_names = ["Q_Cum", "Q_Cum_origin_int", "Q_Cum_origin_dp", "Q_RW",
                "L_Cum", "L_Cum_origin_int", "L_Cum_origin_dp", "L_RW"]

# Create combined dataframe of raw patient data
patients_df = create_patients_df(patient_list, df)

# Create combined dataframe of results of predictions
results_df = create_results_df(dict_list, method_names, patient_list)

# Create combined dataframe with calibration and prediction data for linear and quadratic methods
cal_pred_df = create_cal_pred_df(patient_list, linear_cal_pred, quad_cal_pred)

# Create combined dataframe with each row containing data required for one prediciton
prediction_input_df = create_prediction_input_df(input_dict_list, patient_list, input_method_names)

# Prepare dataframes for export (add type column, sort, reset index)
cal_pred_df, prediction_input_df, results_df = prep_df_for_export(cal_pred_df, 
                                                                  prediction_input_df, 
                                                                  results_df, 
                                                                  patient_list, 
                                                                  method_names)

# Write dataframes to excel
# List of dataframes and sheet names
dfs = [patients_df.set_index('patient'), cal_pred_df.set_index('patient'),\
       prediction_input_df.set_index('patient'), results_df.set_index('patient')]
sheets = ['Patient','Calibration Prediction','Prediction Input', 'Results']    

# Run function
dfs_tabs(dfs, sheets, 'All_Data.xlsx')


# +
##### PLOTTING #####
# FacetGrid Plot by Method

# Subset dataframe with deviation, prediction day, method, patient
plot_df = results_df[['deviation', 'prediction day', 'method', 'patient']]

# Set up lists
hue_order = [patient for patient in patient_list]
order = [num for num in range(int(plot_df['prediction day'].min()),
                                  int(plot_df['prediction day'].max()+1))]
labels = hue_order
colors = sns.color_palette("Paired").as_hex()[:len(labels)]
handles = [patches.Patch(color=col, label=lab) for col, lab in zip(colors, labels)]

# Set up facet grid
g = sns.FacetGrid(plot_df, col_wrap=5, col="method", hue="patient", palette=colors, aspect=1.4)

# Visualise data on grid

g.map(sns.pointplot, "prediction day", "deviation", hue_order=hue_order, order=order)

# Set xtick labels to rotate
g.set_xticklabels(rotation=45)

# Add legend with plt.legend, add_legend has a bug
plt.legend(handles=handles, title="patient", loc="upper left", bbox_to_anchor=(1.25,0.9))


# +
##### PLOTTING #####
# FacetGrid Plot by Patient

# Subset dataframe with deviation, prediction day, method, patient
plot_df = results_df[['deviation', 'prediction day', 'method', 'patient']]

# Set up lists
hue_order = [method for method in method_names]
order = [num for num in range(int(plot_df['prediction day'].min()),
                                  int(plot_df['prediction day'].max()+1))]
labels = hue_order
colors = sns.color_palette("Paired").as_hex()[:len(labels)]
handles = [patches.Patch(color=col, label=lab) for col, lab in zip(colors, labels)]

# Set up facet grid
g = sns.FacetGrid(plot_df, col_wrap=4, col="patient", hue="method", palette=colors, aspect=1.4)

# Visualise data on grid

g.map(sns.pointplot, "prediction day", "deviation", hue_order=hue_order, order=order)

# Set xtick labels to rotate
g.set_xticklabels(rotation=45)

# Add legend with plt.legend, add_legend has a bug
plt.legend(handles=handles, title="method", loc="upper left", bbox_to_anchor=(1.25,0.9))

