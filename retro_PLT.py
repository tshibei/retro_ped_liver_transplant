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

# +
input_file = 'Retrospective Liver Transplant Data.xlsx'
rows_to_skip = 17

# Get list of patients/sheet names
list_of_patients = get_sheet_names(input_file)
list_of_patient_df = []

for patient in list_of_patients:
        
    df = pd.read_excel(input_file, sheet_name=patient, skiprows=rows_to_skip)
    
    # Clean data
    df = clean_data(df, patient)
    
    # Keep ideal data 
    

    # Add patient column
    df['patient'] = patient
    
    df.columns = ['day', 'response', 'dose', 'patient']
    
    list_of_patient_df.append(df)

df = pd.concat(list_of_patient_df)


