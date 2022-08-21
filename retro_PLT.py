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
from analysis import *
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

# +
# %%time
# ~5mins

# Execute CURATE without pop tau
execute_CURATE(pop_tau_string='_by_body_weight_2')


# +
output = 'output_dose_by_body_weight.xlsx'

# Create dataframe for effect of CURATE
minimum_capsule = 0.5
therapeutic_range_upper_limit = 10
therapeutic_range_lower_limit = 8
overprediction_limit = -1.5
underprediction_limit = 2

# -

dat = create_df_for_CURATE_assessment()



a = []
len(a)

# +
output = 'output_dose_by_body_weight.xlsx'

dat = CURATE_could_be_useful()

# Subset RW
dat = dat[dat.method=='L_RW_wo_origin'].reset_index(drop=True)

# Check if recommended doses are less than 0.55mg/kg/day
dat['reasonable_dose'] = True
for i in range(len(dat)):
    dat.loc[i, 'reasonable_dose'] = min(dat.interpolated_dose_8[i], dat.interpolated_dose_9[i], dat.interpolated_dose_10[i]) < 0.85

# Change pred_day to day 
dat = dat.rename(columns={'pred_day':'day'})

# Add original dose column
clean_data = pd.read_excel(output, sheet_name='clean')
combined_data = dat.merge(clean_data[['day', 'patient', 'dose_mg']], how='left', on=['patient', 'day'])

# Declare lists
list_of_patients = []
list_of_body_weight = []

list_of_patients = find_list_of_patients()
list_of_body_weight = find_list_of_body_weight()

# Add body weight column
combined_data['body_weight'] = ""
for j in range(len(combined_data)):
    index_patient = list_of_patients.index(str(combined_data.patient[j]))
    combined_data.loc[j, 'body_weight'] = list_of_body_weight[index_patient]

combined_data['interpolated_dose_8_mg'] = combined_data['interpolated_dose_8'] * combined_data['body_weight']
combined_data['interpolated_dose_9_mg'] = combined_data['interpolated_dose_9'] * combined_data['body_weight']
combined_data['interpolated_dose_10_mg'] = combined_data['interpolated_dose_10'] * combined_data['body_weight']


combined_data
# combined_data[['interpolated_dose_8_mg', 'interpolated_dose_9_mg', 'interpolated_dose_10_mg']]


# +
# number_of_unreasonable_doses = len(dat) - dat.reasonable_dose.sum()

# # Keep reasonable doses
# dat = dat[dat.reasonable_dose==True].reset_index(drop=True)


# Archived code from cell above

# # Find number of wrong range predictions
# number_of_unreliable_predictions = dat['wrong_range'].sum()

# # Keep reliable predictions
# dat = dat[dat.wrong_range==False].reset_index(drop=True)

# # Find number of inaccurate predictions with clinically acceptable prediction error
# number_of_inaccurate_predictions = len(dat) - dat.acceptable_deviation.sum()

# # Keep accurate predictions
# dat = dat[dat.acceptable_deviation==True].reset_index(drop=True)

recommended_dose_mg = [2.5, 2.5, 4.5, 5.5, 5, 5, 2, 2.5, 4.5, 4.5, np.nan, np.nan, 0, 6, 1.5, 2, 2.5, 3.5, 3.5, 2, 0,
                    1.5, 1.5, np.nan, np.nan, np.nan, np.nan, 2.5, 2.5, 0, np.nan, 3, np.nan, 0.5, 0, 0, 2.5, 2.5, 3]

combined_data['recommended_dose_mg'] = recommended_dose_mg

combined_data['diff_dose_mg'] = combined_data['dose_mg'] - combined_data['recommended_dose_mg']
combined_data['abs_diff_dose_mg'] = abs(combined_data['dose_mg'] - combined_data['recommended_dose_mg'])
combined_data['diff_dose_mg_boolean'] = combined_data['abs_diff_dose_mg'] >= 0.5
combined_data['recommended_dose'] = combined_data['recommended_dose_mg'] / combined_data['body_weight']

number_of_similar_dose = len(combined_data) - combined_data.diff_dose_mg_boolean.sum()

# Keep those with diff dose
combined_data = combined_data[combined_data.diff_dose_mg_boolean==True].reset_index(drop=True)

# Count number of non-therapeutic range
number_of_non_therapeutic_range = len(combined_data) - combined_data.within_range.sum()

# Keep non-therapeutic range only
combined_data = combined_data[combined_data.within_range == False].reset_index(drop=True)

combined_data['diff_dose'] = combined_data['recommended_dose'] - combined_data['dose']
combined_data['abs_diff_dose'] = abs(combined_data['dose'] - combined_data['recommended_dose'])

combined_data['CURATE_could_be_useful'] = (combined_data.acceptable_deviation==True) & \
(combined_data.wrong_range==False) & \
    (combined_data.reasonable_dose==True) & \
        (combined_data.within_range==False)

combined_data.diff_dose.astype(float).describe()
# -

dat = pd.read_excel('all_data_including_non_ideal.xlsx', sheet_name='data')
dat.columns

clean = pd.read_excel('output_dose_by_body_weight.xlsx', sheet_name='clean')

dat = clean.copy()
dat.dose.describe()

result = pd.read_excel('output_dose_by_body_weight.xlsx', sheet_name='result')

dat = result.copy()
dat[dat.method=='L_RW_wo_origin']['prediction'].describe()

df = CURATE_could_be_useful()

21/119*100

# +
# %%time
# Perform CV
five_fold_cross_val_results, five_fold_cross_val_results_summary = find_pop_tau_with_CV()
execute_CURATE_and_update_pop_tau_results('CV', five_fold_cross_val_results_summary, five_fold_cross_val_results)

# Perform LOOCV
five_fold_cross_val_results, five_fold_cross_val_results_summary = find_pop_tau_with_LOOCV()
execute_CURATE_and_update_pop_tau_results('LOOCV', five_fold_cross_val_results_summary, five_fold_cross_val_results)

# +
# Profile Generation
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
list_of_body_weight = []

# Create list of body_weight
for i in range(len(list_of_patients)):    
    data = pd.read_excel('Retrospective Liver Transplant Data.xlsx', list_of_patients[i], index_col=None, usecols = "C", nrows=15)
    data = data.reset_index(drop=True)
    list_of_body_weight.append(data['Unnamed: 2'][13])
    
list_of_body_weight = list_of_body_weight[:12]+[8.29]+list_of_body_weight[12+1:]

number_of_patients = 0

for patient in list_of_patients:

    # Create and clean patient dataframe        
    df = pd.read_excel(input_file, sheet_name=patient, skiprows=rows_to_skip)
    df = clean_data(df, patient)
    df = keep_ideal_data(df, patient, list_of_patient_df)
    
    # Change to dose by body weight
    df['dose'] = df['dose'] / list_of_body_weight[number_of_patients]
    
    # Counter for number of patients
    number_of_patients = number_of_patients + 1
    
list_of_patient_df

# +


list_of_body_weight

# +
file_string='all_data_including_non_ideal.xlsx'
plot=True

# Plot individual profiles
dat = pd.read_excel(file_string, sheet_name='clean')

# Create within-range column for color
dat['within_range'] = (dat.response <= 10) & (dat.response >= 8)

# Create low/med/high dose column
dat['dose_range'] = ""
for i in range(len(dat)):
    if dat.dose[i] < 2:
        dat.loc[i, 'dose_range'] = 'Low'
    elif dat.dose[i] < 4:
        dat.loc[i, 'dose_range'] = 'Medium'
    else:
        dat.loc[i, 'dose_range'] = 'High'

# Rename columns and entries
new_dat = dat.copy()
new_dat = new_dat.rename(columns={'within_range':'Tacrolimus Levels'})
new_dat['Tacrolimus Levels'] = new_dat['Tacrolimus Levels'].map({True:'Therapeutic Range', False: 'Non-therapeutic Range'})
new_dat = new_dat.rename(columns={'dose_range':'Dose range', 'day':'Day'})
new_dat['patient'] = new_dat['patient'].map({84:1, 114:2, 117:3, 118:4, 120:5, 121:6, 122:7,
                                            123:8, 125:9, 126:10, 129:11, 130:12, 131:13, 132:14,
                                            133:15, 138:16})

if plot == True:

    # Plot dose vs response
    sns.set(font_scale=1.2, rc={"figure.figsize": (16,10), "xtick.bottom" : True, "ytick.left" : True}, style='white')

    plot = plt.scatter(new_dat.dose, new_dat.response, c=new_dat.Day, cmap=sns.cubehelix_palette(as_cmap=True))
    plt.clf()
    cbar = plt.colorbar(plot)
    cbar.ax.tick_params(labelsize=20) 

    plt.savefig('colorbar.png', dpi=500, facecolor='w', bbox_inches='tight')

#     g = sns.relplot(data=new_dat, x='dose', y='response', hue='Day', col='patient', col_wrap=4, style='Dose range',
#             height=3, aspect=1,s=80)

#     # Add gray region for therapeutic range
#     for ax in g.axes:
#         ax.axhspan(8, 10, facecolor='grey', alpha=0.2)
        
#     g.set_titles('Patient {col_name}')
#     g.set_ylabels('Tacrolimus level (ng/ml)')
#     g.set_xlabels('Dose')
#     g.set(yticks=np.arange(0,math.ceil(max(new_dat.response)),4),
#         xticks=np.arange(0, max(new_dat.dose+1), step=1))

#     plt.savefig('indiv_pt_profile_by_dose.png', dpi=500, facecolor='w', bbox_inches='tight')
# -


df = effect_of_CURATE_RW()

dat = df.copy()

df = effect_of_CURATE_RW()

# +
dat = df.copy()

# Keep within_range only
dat_SOC_within_range = dat[dat.within_range==True].reset_index(drop=True)

# Find first day to achieve therapeutic range in SOC
SOC = dat_SOC_within_range.groupby('patient')['Day'].first().to_frame()

SOC_achieve_TR_in_first_week = SOC <= 7

# Keep those that are 1) unaffected by in TR and 2) improve to TR
dat_CURATE_within_range = dat[(dat['Effect of CURATE.AI-assisted dosing']=='Unaffected, remain as therapeutic range') |
                             (dat['Effect of CURATE.AI-assisted dosing']=='Improve to therapeutic range')].reset_index(drop=True)

# Find first day to achieve therapeutic range in CURATE
CURATE = dat_CURATE_within_range.groupby('patient')['Day'].first().to_frame()

CURATE_achieve_TR_in_first_week = CURATE <= 7

# Create dataframe with vales
plot_data = pd.Series({'Standard of care\ndosing': SOC_achieve_TR_in_first_week.sum()[0], 'CURATE.AI-assisted\ndosing':CURATE_achieve_TR_in_first_week.sum()[0]}).to_frame().reset_index()
plot_data.columns = ['Dosing', 'Patients that achieved therapeutic range in first week']

sns.set(font_scale=1.2, rc={"figure.figsize": (4,5), "xtick.bottom":True, "ytick.left":True}, style='white')
fig, ax = plt.subplots()
ax.bar(plot_data['Dosing'], plot_data['Patients that achieved therapeutic range in first week'], width=0.5, color=['y', 'm'])
sns.despine()
plt.ylabel('Patients that achieved\ntherapeutic range in first week')

# CURATE_achieve_TR_in_first_week.sum()[0]

# combined = SOC.merge(CURATE, how='left', on='patient')
# combined.columns=['SOC', 'CURATE']

# combined = combined.stack().to_frame().reset_index()
# combined.columns = ['patient', 'Dosing', 'Days to first therapeutic range']
# combined['Dosing'] = combined['Dosing'].replace({'SOC':'Standard of care', 'CURATE':'CURATE.AI-assisted'})

# sns.set(font_scale=1.2, rc={"figure.figsize": (5,5), "xtick.bottom":True, "ytick.left":True}, style='white')
# g = sns.boxplot(x="Dosing", y="Days to first therapeutic range", data=combined, width=0.5, palette=['#ccb974','#8172b3'])
# sns.despine()
# -


