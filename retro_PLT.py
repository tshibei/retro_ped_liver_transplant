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

# +
# %%time
# ~18mins

# Execute CURATE without pop tau
execute_CURATE()


# +
# %%time
# Perform CV
five_fold_cross_val_results, five_fold_cross_val_results_summary = find_pop_tau_with_CV()
execute_CURATE_and_update_pop_tau_results('CV', five_fold_cross_val_results_summary, five_fold_cross_val_results)

# Perform LOOCV
five_fold_cross_val_results, five_fold_cross_val_results_summary = find_pop_tau_with_LOOCV()
execute_CURATE_and_update_pop_tau_results('LOOCV', five_fold_cross_val_results_summary, five_fold_cross_val_results)
# -
df = pd.read_excel('output (with pop tau by LOOCV).xlsx', sheet_name='result')

# +
from matplotlib import colors

dat = df.copy()

# Subset L_RW_wo_origin and patient 118
dat = dat[(dat.method=='L_RW_wo_origin') &  (dat.patient==118)]

dat = dat[['patient', 'method', 'pred_day', 'dose', 'response', 'coeff_1x', 'coeff_0x', 'prediction', 'deviation', 'fit_dose_1', 'fit_dose_2', 'fit_response_1', 'fit_response_2', 'day_1', 'day_2']].reset_index(drop=True)

# Interpolate to find percentage of possible dosing events for when prediction and observed response are outside range
for i in range(len(dat)):
    # Create function
    coeff = dat.loc[i, 'coeff_1x':'coeff_0x'].apply(float).to_numpy()
    coeff = coeff[~np.isnan(coeff)]
    p = np.poly1d(coeff)
    x = np.linspace(0, max(dat.dose)+ 2)
    y = p(x)
    order = y.argsort()
    y = y[order]
    x = x[order]

    dat.loc[i, 'interpolated_dose_8'] = np.interp(8, y, x)
    dat.loc[i, 'interpolated_dose_9'] = np.interp(9, y, x)
    dat.loc[i, 'interpolated_dose_10'] = np.interp(10, y, x)
    
# Create column to find points that outperform, benefit, or do not affect SOC
dat['effect_on_SOC'] = 'none'
dat['predict_range'] = 'therapeutic'
dat['response_range'] = 'therapeutic'
dat['prediction_error'] = 'acceptable'
dat['diff_dose'] = '>0.5'
for i in range(len(dat)):

    if (dat.prediction[i] > 10) or (dat.prediction[i] < 8):
        dat.loc[i,'predict_range'] = 'non-therapeutic'
        if (dat.response[i] > 10) or (dat.response[i] < 8):
            dat.loc[i,'response_range'] = 'non-therapeutic'
            if (round(dat.deviation[i],2) > -2) and (round(dat.deviation[i],2) < 1.5):
                if (abs(dat.interpolated_dose_8[i] - dat.dose[i]) or abs(dat.interpolated_dose_9[i] - dat.dose[i]) or abs(dat.interpolated_dose_10[i] - dat.dose[i])) > 0.5:
                    dat.loc[i, 'effect_on_SOC'] = 'outperform'
        elif (dat.response[i] <= 10) and (dat.response[i] >= 8):
                dat.loc[i, 'effect_on_SOC'] = 'worsen'

# Subset columns
dat = dat[['pred_day', 'effect_on_SOC', 'fit_dose_1', 'fit_dose_2', 'fit_response_1', 'fit_response_2', 'day_1', 'day_2']]

# Stack columns to fit dataframe for plotting
df_fit_dose = dat[['pred_day', 'effect_on_SOC', 'fit_dose_1', 'fit_dose_2']]
df_fit_dose = df_fit_dose.set_index(['pred_day', 'effect_on_SOC'])
df_fit_dose = df_fit_dose.stack().reset_index()
df_fit_dose.columns = ['pred_day', 'effect_on_SOC', 'fit_dose', 'x']
df_fit_dose = df_fit_dose.reset_index()

df_fit_response = dat[['pred_day', 'effect_on_SOC', 'fit_response_1', 'fit_response_2']]
df_fit_response = df_fit_response.set_index(['pred_day', 'effect_on_SOC'])
df_fit_response = df_fit_response.stack().reset_index()
df_fit_response.columns = ['pred_day', 'effect_on_SOC', 'fit_response', 'y']
df_fit_response = df_fit_response.reset_index()

df_day = dat[['pred_day', 'effect_on_SOC', 'day_1', 'day_2']]
df_day = df_day.set_index(['pred_day', 'effect_on_SOC'])
df_day = df_day.stack().reset_index()
df_day.columns = ['pred_day', 'effect_on_SOC', 'day_num', 'day']
df_day = df_day.reset_index()

combined_df = df_fit_dose.merge(df_fit_response, how='left', on=['index', 'pred_day', 'effect_on_SOC'])
combined_df = combined_df.merge(df_day, how='left', on=['index', 'pred_day', 'effect_on_SOC'])

# Plot
sns.set(font_scale=1.2, rc={"figure.figsize": (16,10), "xtick.bottom":True, "ytick.left":True},
        style='white')
g = sns.lmplot(data=combined_df, x='x', y='y', hue='pred_day', ci=None, legend=False)

ec = colors.to_rgba('black')
ec = ec[:-1] + (0.3,)

for i in range(combined_df.shape[0]):
    plt.text(x=combined_df.x[i]+0.3,y=combined_df.y[i]+0.3,s=int(combined_df.day[i]), 
      fontdict=dict(color='black',size=13),
      bbox=dict(facecolor='white', ec='black', alpha=0.5, boxstyle='circle'))
    
    plt.text(x=0+0.3,y=10.4+0.3,s=12, 
      fontdict=dict(color='black',size=13),
      bbox=dict(facecolor='white', ec=ec, boxstyle='circle'))
    
    plt.text(x=0+0.3,y=8.7+0.3,s=14, 
      fontdict=dict(color='black',size=13),
      bbox=dict(facecolor='white', ec=ec, boxstyle='circle'))

plt.legend(bbox_to_anchor=(1.06,0.5), loc='center left', title='Day of Prediction', frameon=False)
plt.xlabel('Tacrolimus dose (mg)')
plt.ylabel('Tacrolimus level (ng/ml)')
plt.axhspan(8, 10, facecolor='grey', alpha=0.2)

# Add data point of day 12 and day 14
plt.plot(0, 10.4, marker="o", markeredgecolor="black", markerfacecolor="white")
plt.plot(0, 8.7, marker="o", markeredgecolor="black", markerfacecolor="white")
plt.savefig('patient_118_RW_profiles.png', dpi=500, facecolor='w', bbox_inches='tight')


# -

method_dat, method_string = CURATE_simulated_results_PPM_RW()

dat = method_dat.copy()
df_RW = dat[1]
df_RW[df_RW.patient==118]

df = CURATE_could_be_useful()

# +
dat = df.copy()

# Subset chosen results
dat = dat[(dat.wrong_range==False) & (dat.acceptable_deviation==True)]

# Subset columns
dat = dat[['method']]

# +
dat = CURATE_could_be_useful()

method_string = ['PPM', 'RW']
method_dat = []

for j in range(len(method_string)):

    # Subset selected PPM/RW method
    dat = dat[dat['method']==('L_' + method_string[j] + '_wo_origin')]

    # Create column for adapted within range to indicate if data point
    # could have been within range if augmented by CURATE
    dat['adapted_within_range'] = dat.within_range
    dat = dat.reset_index()

    for i in range(len(dat)):
        if (dat.CURATE_could_be_useful[i]==True):
            dat.loc[i, 'adapted_within_range'] = 'Potentially True with CURATE_' + method_string[j]
        elif (dat.within_range[i]==True and (dat.wrong_range[i]==True or dat.acceptable_deviation[i]==False)):
            dat.loc[i, 'adapted_within_range'] = 'Potentially False with CURATE_' + method_string[j]
        else:
            dat.loc[i, 'adapted_within_range'] = 'CURATE_no_impact'

    # Subset columns
    dat = dat[['pred_day', 'patient', 'adapted_within_range']]

    # Rename columns
    dat.columns = ['day', 'patient', 'adapted_within_range']

    # Only keep those that are affected by SOC
    dat = dat[dat.adapted_within_range != 'CURATE_no_impact']
    dat = dat[['day', 'patient', 'adapted_within_range']]

    # Import data with all data including non-ideal data
    dat_all_data = indiv_profiles_all_data(plot=False)

    # Merge both dataframes
    combined_dat = dat_all_data.merge(dat, how='left', on=['patient', 'day'])
    combined_dat.loc[combined_dat['adapted_within_range'].isnull(),'adapted_within_range'] = \
    combined_dat['within_range']
    combined_dat['adapted_within_range'] = combined_dat['adapted_within_range'].astype(str)

    # Rename adapted_within_range
    for i in range(len(combined_dat)):
        if combined_dat.adapted_within_range[i] == 'Potentially True with CURATE_' + method_string[j]:
            combined_dat.loc[i, 'adapted_within_range'] = 'True (' + method_string[j] + '_assisted)'
        elif combined_dat.adapted_within_range[i] == 'Potentially False with CURATE_' + method_string[j]:
            combined_dat.loc[i, 'adapted_within_range'] = 'False (' + method_string[j] + '_assisted)'

    # Plot
    sns.set(font_scale=1.2, rc={'figure.figsize':(16,10)})
    sns.set_style('white')
    hue_order = ['True', 'False', 'True (' + method_string[j] + '_assisted)', 'False (' + method_string[j] + '_assisted)']
    palette = [sns.color_palette()[1], sns.color_palette()[0], sns.color_palette()[2],\
              sns.color_palette()[3]]

    g = sns.relplot(data=combined_dat, x='day', y='response', hue='adapted_within_range',\
                    hue_order=hue_order, col='patient', palette=palette,\
                    col_wrap=4, style='dose_range', height=1.5, aspect=1.5, s=60)

    g.map(plt.axhline, y=10, ls='--', c='black')
    g.map(plt.axhline, y=8, ls='--', c='black')

    plt.savefig('indiv_pt_profile_adapted_' + method_string[j] + '.png', dpi=500, facecolor='w', bbox_inches='tight')

    method_dat.append(combined_dat)
# -

dat = read_file_and_remove_unprocessed_pop_tau()

# +
df = method_dat.copy()
final_df, CURATE_may_help = CURATE_assisted_result_distribution(df, 'PPM')
# final_df

CURATE_may_help
# -

dat = df.copy()
dat

df = CURATE_could_be_useful()

dat, string = CURATE_simulated_results_PPM_RW()

method_dat = dat.copy()
method_string = string.copy()
final_df_list = CURATE_assisted_result_distribution(method_dat, method_string)

