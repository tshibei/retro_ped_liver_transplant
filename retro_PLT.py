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
import warnings
warnings.simplefilter('ignore', np.RankWarning)
from scipy.optimize import curve_fit
import matplotlib.patches as patches
from scipy.optimize import curve_fit
from openpyxl import load_workbook
pd.set_option('display.max_rows', None)

# +
input_file = 'Retrospective Liver Transplant Data.xlsx'
rows_to_skip = 17

# Get list of patients/sheet names
list_of_patients = get_sheet_names(input_file)

# Define lists
list_of_patient_df = []
list_of_cal_pred_df = []
list_of_result_df = []

patients_to_exclude_linear = []
patients_to_exclude_quad = []

for patient in list_of_patients:
        
    df = pd.read_excel(input_file, sheet_name=patient, skiprows=rows_to_skip)
    
    df = clean_data(df, patient)
    df = keep_ideal_data(df, patient, list_of_patient_df)
        
    # Choose and keep data for calibration and efficacy-driven dosing
    cal_pred_linear, patients_to_exclude_linear = cal_pred_data(df, patient, patients_to_exclude_linear, 1)
    cal_pred_quad, patients_to_exclude_quad = cal_pred_data(df, patient, patients_to_exclude_quad, 2)
    
    # Keep patient data with sufficient dose-response pairs and predictions
    cal_pred, list_of_cal_pred_df = keep_target_patients(patient, patients_to_exclude_linear, patients_to_exclude_quad, 
                                                     cal_pred_linear, cal_pred_quad, list_of_cal_pred_df)

    # Prepare dataframe for prediction
    # Create result DataFrame
#     max_count_input = len(cal_pred[cal_pred['type'] =='linear'])
#     print(f"{patient}: max_count_input {max_count_input} | {cal_pred}")
#     col_names = ['patient', 'method', 'pred_day'] + \
#                 ['fit_dose_' + str(i) for i in range(1, max_count_input + 1)] + \
#                 ['fit_response_' + str(i) for i in range(1, max_count_input + 1)] + \
#                 ['dose', 'response', 'prev_coeff_2x', 'prev_coeff_1x', 'prev_coeff_0x',\
#                  'prev_deviation', 'coeff_2x', 'coeff_1x', 'coeff_0x', 'prediction', 'deviation',
#                  'abs_deviation']
#     result = pd.DataFrame(columns=col_names)
    
#     j = 0
#     if patient not in patients_to_exclude_linear:
#         deg = 1
        
#         # Prepare dataframe for L_Cum_wo_origin
#         for i in range(deg + 1, len(cal_pred[cal_pred['type']=='linear'])):
#             result.loc[j, 'patient'] = cal_pred.loc[i, 'patient']
#             result.loc[j, 'method'] = 'L_Cum_wo_origin'
#             result.loc[j, 'pred_day'] = cal_pred.loc[i, 'day']
#             # result.loc[j, 'fit_dose_1':'fit_dose_' + str(i + deg + 1)] = cal_pred.loc[0:i-1, 'dose']
#             result.loc[j, 'fit_dose_1':'fit_dose_' + str(i)] = cal_pred.loc[0:i-1, 'dose'].to_numpy()
#             result.loc[j, 'fit_response_1':'fit_response_' + str(i)] = cal_pred.loc[0:i-1, 'response'].to_numpy()
#             result.loc[j, 'dose'] = cal_pred.loc[i, 'dose']
#             result.loc[j, 'response'] = cal_pred.loc[i, 'response']
#             j = j + 1
            
        # list_of_result_df.append(result)
            
        # print(result)


# Print patients to exclude        
patients_to_exclude_linear = sorted(set(patients_to_exclude_linear))
patients_to_exclude_quad = sorted(set(patients_to_exclude_quad))
print(f"Patients to exclude for linear methods: {patients_to_exclude_linear}")
print(f"Patients to exclude for quad methods: {patients_to_exclude_quad}")

# Join dataframes from individual patients
df = pd.concat(list_of_patient_df)
df.reset_index(inplace=True, drop=True)
cal_pred = pd.concat(list_of_cal_pred_df)
# result_df = pd.concat(list_of_result_df)

# result_df.columns

# -


cal_pred

# +

result



# -
a = pd.DataFrame(columns=['col1', 'col2', 'COL1'])
a.loc[0,:] = [1,2,3]
b = pd.DataFrame(columns=['col1', 'col2', 'col3', 'COL1'])
b.loc[0,:] = [1,2,3,4]
pd.concat([a, b])


        

a = pd.DataFrame(columns=['a', 'b', 'c'])
a.loc[0,'a':'c'] = 5
a.loc[1, 'a': 'c'] = 4
a.loc[0:1-1, 'a']


