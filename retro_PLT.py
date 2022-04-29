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


def cal_pred_data(df, patient, patients_to_exclude, deg):
    """
    Find calibration points and combine calibration data with the rest of data,
    to perform CURATE methods later.
    
    Input: 
    df - dataframe
    patient - string of patient number
    deg - degree of fitting polynomial (1 for linear, 2 for quadratic)

    Output: 
    Dataframe of calibration data and data for subsequent predictions.
    Print patients with insufficient data for calibration and <3 predictions

    """
    # Create index column
    df = df.reset_index(drop=False) 

    # Create boolean column to check if tac dose diff from next row
    df['diff_from_next'] = \
            (df['dose'] != df['dose'].shift(-1))

    # Find indexes of last rows of first 2 unique doses
    last_unique_doses_idx = [i for i, x in enumerate(df.diff_from_next) if x]

    # Create boolean column to check if tac dose diff from previous row
    df['diff_from_prev'] = \
            (df['dose'] != df['dose'].shift(1))

    # Find indexes of first rows of third unique dose
    first_unique_doses_idx = [i for i, x in enumerate(df.diff_from_prev) if x]

    # The boolean checks created useless index, diff_from_next and diff_from_prev columns,
    # drop them
    df = df.drop(['index', 'diff_from_next', 'diff_from_prev'], axis=1)

    # Combine calibration and prediction rows
    cal_pred = pd.DataFrame()
    
    # Do for quadratic method
    if deg == 2:
        if df['dose'].nunique() > 2:
            # If third cal point is the same as first 2 cal points, keep looking
            first_cal_dose = df['dose'][last_unique_doses_idx[0]]
            second_cal_dose = df['dose'][last_unique_doses_idx[1]]
            n = 2
            for i in range(n, len(df)+1):
                third_cal_dose = df['dose'][first_unique_doses_idx[n]]
                if (third_cal_dose == first_cal_dose) | (third_cal_dose == second_cal_dose):
                    n = n + 1

            first_cal_point = pd.DataFrame(df.iloc[last_unique_doses_idx[0],:]).T
            second_cal_point = pd.DataFrame(df.iloc[last_unique_doses_idx[1],:]).T
            third_cal_point = pd.DataFrame(df.iloc[first_unique_doses_idx[n],:]).T
            rest_of_data = df.iloc[first_unique_doses_idx[n]+1:,:]
            cal_pred = pd.concat([first_cal_point, second_cal_point, third_cal_point, 
                                        rest_of_data]).reset_index(drop=True)

    # Do for linear method
    if deg == 1:
        if df['dose'].nunique() > 1:
            first_cal_point = pd.DataFrame(df.iloc[last_unique_doses_idx[0],:]).T
            second_cal_point = pd.DataFrame(df.iloc[first_unique_doses_idx[1],:]).T
            rest_of_data = df.iloc[first_unique_doses_idx[1]+1:,:]
            cal_pred = pd.concat([first_cal_point, second_cal_point, 
                                        rest_of_data]).reset_index(drop=True)
        else:
            patients_to_exclude.append(str(patient))
            print(patient, ": Insufficient unique dose-response pairs for linear calibration!")

    # Print error msg if number of predictions is less than 3
    if df['dose'].nunique() < 3:
        pass # there are insufficient data for calibration already, don't need
             # this error msg
    elif len(cal_pred) - (deg + 1) < 3:
        patients_to_exclude.append(str(patient))
        if deg == 1:
            error_string = '(for linear)'
        else:
            error_string = '(for quadratic)'
        print(patient, ": No. of predictions ", error_string," is <3: ", len(cal_pred) - (deg + 1))  
    
    # Add "type" column
    cal_pred['type'] = ""
    if deg == 1:
        cal_pred['type'] = 'linear'
    else:
        cal_pred['type'] = 'quadratic'

    return cal_pred, patients_to_exclude



# +
input_file = 'Retrospective Liver Transplant Data.xlsx'
rows_to_skip = 17

# Get list of patients/sheet names
list_of_patients = get_sheet_names(input_file)
list_of_patient_df = []
list_of_cal_pred_df = []
patients_to_exclude = []

for patient in list_of_patients:
        
    df = pd.read_excel(input_file, sheet_name=patient, skiprows=rows_to_skip)
    
    df = clean_data(df, patient)
    df = keep_ideal_data(df)
    df['patient'] = patient
    df.columns = ['day', 'response', 'dose', 'patient']
    list_of_patient_df.append(df)
    
    # Pick rows for prediction
    cal_pred_linear, patients_to_exclude = cal_pred_data(df, patient, patients_to_exclude, 1)
    cal_pred_quad, patients_to_exclude = cal_pred_data(df, patient, patients_to_exclude, 2)
    cal_pred = pd.concat([cal_pred_linear, cal_pred_quad])
    list_of_cal_pred_df.append(cal_pred)
    
df = pd.concat(list_of_patient_df)
df.reset_index(inplace=True, drop=True)
cal_pred = pd.concat(list_of_cal_pred_df)
cal_pred



