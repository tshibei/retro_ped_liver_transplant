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
    
    # Create and clean patient dataframe        
    df = pd.read_excel(input_file, sheet_name=patient, skiprows=rows_to_skip)
    df = clean_data(df, patient)
    df = keep_ideal_data(df, patient, list_of_patient_df)
        
    # Select data for calibration and efficacy-driven dosing
    cal_pred_linear, patients_to_exclude_linear = cal_pred_data(df, patient, patients_to_exclude_linear, 1)
    cal_pred_quad, patients_to_exclude_quad = cal_pred_data(df, patient, patients_to_exclude_quad, 2)
    
    # Keep patients with sufficient dose-response pairs and predictions for each method
    cal_pred, list_of_cal_pred_df = keep_target_patients(patient, patients_to_exclude_linear, patients_to_exclude_quad, 
                                                     cal_pred_linear, cal_pred_quad, list_of_cal_pred_df)   
    
    # Apply methods
    list_of_result_df = apply_methods(cal_pred, patient, patients_to_exclude_linear, patients_to_exclude_quad,
              cal_pred_linear, cal_pred_quad, list_of_result_df)

# Print patients to exclude        
patients_to_exclude_linear = sorted(set(patients_to_exclude_linear))
patients_to_exclude_quad = sorted(set(patients_to_exclude_quad))
print(f"Patients to exclude for linear methods: {patients_to_exclude_linear}")
print(f"Patients to exclude for quad methods: {patients_to_exclude_quad}")

# Join dataframes from individual patients
df = pd.concat(list_of_patient_df)
df.reset_index(inplace=True, drop=True)
cal_pred = pd.concat(list_of_cal_pred_df)
result_df = pd.concat(list_of_result_df)
result_df = format_result_df(cal_pred, result_df)
# -


list().pop()

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


