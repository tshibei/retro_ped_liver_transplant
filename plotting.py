import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
import math
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.pyplot import cm
from matplotlib.patches import Patch
import matplotlib as mpl
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages
from openpyxl import load_workbook
import sys
from scipy import stats
from scipy.stats import mannwhitneyu
from scipy.stats import levene
from scipy.stats import wilcoxon
from scipy.stats import bartlett
from statistics import *
from scipy import interpolate

from implement_CURATE import *

# Define file names
result_file_total = 'CURATE_results.xlsx'
result_file_evening = 'CURATE_results_evening_dose.xlsx'
raw_data_file = 'Retrospective Liver Transplant Data - edited.xlsx'
all_data_file_total = 'all_data_total.xlsx'
all_data_file_evening = 'all_data_evening.xlsx'

# Define clinically relevant parameters
low_dose_upper_limit = 2
medium_dose_upper_limit = 4
low_dose_upper_limit_BW = 0.3
medium_dose_upper_limit_BW = 0.6
overprediction_limit = -1.5
underprediction_limit = 2
max_dose_recommendation = 8
min_dose_recommendation = 0
max_dose_recommendation_BW = 0.85
minimum_capsule = 0.5
therapeutic_range_upper_limit = 10
therapeutic_range_lower_limit = 8
dosing_strategy_cutoff = 0.4
acceptable_tac_upper_limit = 12
acceptable_tac_lower_limit = 6.5

# Patient population
def percentage_of_pts_that_reached_TR_per_dose_range(all_data_file=all_data_file_total):
    """Find percentage of patients that reached TR at the same dose range."""
    # Filter doses that reached TR
    df = pd.read_excel(all_data_file_total)
    df = df[df['dose'].notna()].reset_index(drop=True)
    for i in range(len(df)):
        if (df.response[i] >= therapeutic_range_lower_limit) & (df.response[i] <= therapeutic_range_upper_limit):
            df.loc[i, 'TR'] = True
        else:
            df.loc[i, 'TR'] = False

    df = df[df.TR==True].reset_index(drop=True)

    # Dose range grouped by patients
    for i in range(len(df)):
        if df.dose[i] < low_dose_upper_limit:
            df.loc[i, 'dose_range'] = 'low'
        elif df.dose[i] < medium_dose_upper_limit:
            df.loc[i, 'dose_range'] = ' medium'
        else:
            df.loc[i, 'dose_range'] = 'high'

    # Dose range that reached TR
    df = df.groupby('patient')['dose_range'].unique().astype(str).reset_index(name='dose_range_in_TR')

    # Percentage of patients by TR dose range
    df = df.groupby('dose_range_in_TR')['dose_range_in_TR'].apply(lambda x: x.count()/len(df.patient.unique())*100).reset_index(name='perc_pts')

    # Format percentage
    df['perc_pts'] = df['perc_pts'].apply(lambda x: "{:.2f}".format(x))

    return df
    
def patient_population_values():
    """
    Print out results of the following into a text file 'patient_population_values.txt'
    1. Response
    2. % of days within therapeutic range
    3. % of participants that reached therapeutic range within first week
    4. Day where patient first achieved therapeutic range
    5. Dose administered by mg
    6. Dose administered by body weight
    """
    original_stdout = sys.stdout
    with open('patient_population_values.txt', 'w') as f:
        sys.stdout = f
        
        data = fig_2_TTL_over_time(plot=False)

        # 1. Response
        result_and_distribution(data.response, '1. Response')

        # 2. % of days within therapeutic range

        # Drop rows where response is NaN
        data = data[data.response.notna()].reset_index(drop=True)

        # Add therapeutic range column
        for i in range(len(data)):
            if (data.response[i] >= therapeutic_range_lower_limit) & (data.response[i] <= therapeutic_range_upper_limit):
                data.loc[i, 'therapeutic_range'] = True
            else:
                data.loc[i, 'therapeutic_range'] = False

        perc_therapeutic_range = data.groupby('patient')['therapeutic_range'].apply(lambda x: x.sum()/x.count()*100)
        perc_therapeutic_range = perc_therapeutic_range.to_frame().reset_index()

        # Result and distribution
        result_and_distribution(perc_therapeutic_range.therapeutic_range, '2. % of days within therapeutic range')

        # 3. % of participants that reached therapeutic range within first week
        first_week_df = data.copy()
        first_week_df = first_week_df[first_week_df['Tacrolimus trough levels (TTL)']=='Therapeutic range'].reset_index(drop=True)
        first_week_df = (first_week_df.groupby('patient')['Day'].first() <= 7).to_frame().reset_index()
        result = first_week_df.Day.sum()/first_week_df.Day.count()*100

        print(f'3. % of participants that reached therapeutic range within first week:\n{result:.2f}%,\
        {first_week_df.Day.sum()} out of 16 patients\n')

        # 4. Day where patient first achieved therapeutic range
        first_TR_df = data.copy()
        first_TR_df = first_TR_df[first_TR_df['Tacrolimus trough levels (TTL)']=='Therapeutic range'].reset_index(drop=True)
        first_TR_df = first_TR_df.groupby('patient')['Day'].first().to_frame().reset_index()

        # Result and distribution
        result_and_distribution(first_TR_df.Day, '4. Day where patient first achieved therapeutic range')

        # 5. Dose administered by mg
        dose_df = data.copy()
        result_and_distribution(dose_df.dose, '5. Dose administered')

        # 6. Dose administered by body weight
        dose_df = data.copy()
        result_and_distribution(dose_df.dose_BW, '6. Dose administered by body weight')

    sys.stdout = original_stdout

# Fig 2
def fig_2_TTL_over_time(file_string=all_data_file_total, plot=False, dose='total'):
    """Scatter plot of inidividual profiles, longitudinally, and response vs dose"""
    
    if dose == 'total':
        file_string = all_data_file_total
    else:
        file_string = all_data_file_evening

    # Plot individual profiles
    dat = pd.read_excel(file_string)

    # Create within-range column for color
    dat['within_range'] = (dat.response <= therapeutic_range_upper_limit) & (dat.response >= therapeutic_range_lower_limit)

    # Create low/med/high dose column
    dat['dose_range'] = ""
    for i in range(len(dat)):
        if np.isnan(dat.dose[i]):
             dat.loc[i, 'dose_range'] = 'Unavailable'
        elif dat.dose[i] < low_dose_upper_limit:
            dat.loc[i, 'dose_range'] = 'Low'
        elif dat.dose[i] < medium_dose_upper_limit:
            dat.loc[i, 'dose_range'] = 'Medium'
        else:
            dat.loc[i, 'dose_range'] = 'High'

    # Rename columns and entries
    new_dat = dat.copy()
    new_dat = new_dat.rename(columns={'within_range':'Tacrolimus trough levels (TTL)'})
    new_dat['Tacrolimus trough levels (TTL)'] = new_dat['Tacrolimus trough levels (TTL)'].map({True:'Therapeutic range', False: 'Non-therapeutic range'})
    new_dat = new_dat.rename(columns={'dose_range':'Dose range', 'day':'Day'})
    new_dat['patient'] = new_dat['patient'].map({84:1, 114:2, 117:3, 118:4, 120:5, 121:6, 122:7,
                                                123:8, 125:9, 126:10, 129:11, 130:12, 131:13, 132:14,
                                                133:15, 138:16})

    if plot == True:

        # Add fake row with empty data under response to structure legend columns
        new_dat.loc[len(new_dat.index)] = [2, 5, 0.5, 1, True, "", 1, "", "Low"]
        new_dat.loc[len(new_dat.index)] = [2, 5, 0.5, 1, True, "", 1, " ", "Low"]
        
        # Plot tac levels by day
        sns.set(font_scale=1.5, rc={"figure.figsize": (16,10), "xtick.bottom" : True, "ytick.left" : True}, style='white')

        g = sns.relplot(data=new_dat, x='Day', y='response', hue='Tacrolimus trough levels (TTL)', col='patient', col_wrap=4, style='Dose range',
                height=3, aspect=1,s=100, palette=['tab:blue','tab:orange','white','white'], 
                style_order=['Low', 'Medium', 'High', 'Unavailable'], zorder=2)

        # g = sns.relplot(data=new_dat[new_dat['Dose range']=='Low'], x='Day', y='response', hue='TTL', col='patient', col_wrap=4, style='o',
        # height=3, aspect=1,s=100, palette=['tab:blue','tab:orange','white','white'])
        
        # Add gray region for therapeutic range
        for ax in g.axes:
            ax.axhspan(therapeutic_range_lower_limit, therapeutic_range_upper_limit, facecolor='grey', alpha=0.2, zorder=1)
        
        g.set_titles('Patient {col_name}')
        g.set_ylabels('TTL (ng/ml)')
        g.set(yticks=np.arange(0,math.ceil(max(new_dat.response)),4),
            xticks=np.arange(0, max(new_dat.Day+2), step=5))
        
        # Move legend below plot
        sns.move_legend(g, 'center', bbox_to_anchor=(0.18,-0.08), ncol=2)
        legend_elements = [Patch(facecolor='grey', edgecolor='grey',
                            label='Region within therapeutic range', alpha=.2)]
        legend2 = plt.legend(handles=legend_elements, bbox_to_anchor=(-1.5,-0.42), loc='upper left', frameon=False)

        plt.savefig('TTL_over_time_' + dose + '.png', dpi=1000, facecolor='w', bbox_inches='tight')
        
        # Remove fake row before end of function
        new_dat = new_dat[:-1]

    return new_dat

# Technical performance metrics
def technical_performance_metrics(result_file=result_file_total, dose='total'):
    """
    Print the following technical performance metrics
    1. Prediction erorr
    2. Absolute prediction error
    3. RMSE
    4. LOOCV
    """
    if dose == 'total':
        result_file = result_file_total
    else:
        result_file = result_file_evening

    original_stdout = sys.stdout
    with open('technical_perf_metrics_'+ dose +'.txt', 'w') as f:
        sys.stdout = f

        df = pd.read_excel(result_file, sheet_name='result')

        # Subset method
        df = df[df.method=='L_RW_wo_origin'].reset_index(drop=True)

        # 1. Prediction error
        result_and_distribution(df.deviation, '1. Prediction error')

        print('*for comparison with non-parametric abs prediction error,')
        median_IQR_range(df.deviation)

        # 2. Absolute prediction error
        result_and_distribution(df.abs_deviation, '2. Absolute prediction error')

        # 3. RMSE
        RMSE = (math.sqrt(mean_squared_error(df.response, df.prediction)))
        print(f'3. RMSE: {RMSE:.2f}\n')

        # 4. LOOCV
        print('4. LOOCV\n')
        LOOCV_all_methods(dose=dose)
        experiment = pd.read_excel('LOOCV_results_'+dose+'.xlsx', sheet_name='Experiments')

        experiment = experiment[experiment.method=='L_RW_wo_origin']
        result_and_distribution(experiment.train_median, 'Training set LOOCV')
        result_and_distribution(experiment.test_median, 'Test set LOOCV')
        median_IQR_range(experiment.test_median)
        print(f'Mann whitney u: {mannwhitneyu(experiment.test_median, experiment.train_median)}')

        ## Compare medians between training and test sets

    sys.stdout = original_stdout

# Clinically relevant performance metrics
def clinically_relevant_performance_metrics(result_file=result_file_total, dose='total'):
    """Clinically relevant performance metrics. 
    Find the percentage of predictions within clinically acceptable prediction error,
    percentage of overpredictions, and percentage of underpredictions. Print the results
    into a text file.
    """
    # Uncomment to write output to txt file
    # file_path = 'Clinically relevant performance metrics.txt'
    # sys.stdout = open(file_path, "w")
    if dose == 'total':
        result_file = result_file_total
    else:
        result_file = result_file_evening

    original_stdout = sys.stdout
    with open('clinically_relevant_perf_metrics_'+ dose +'.txt', 'w') as f:
        sys.stdout = f

        df = pd.read_excel(result_file, sheet_name='result')
        df = df[df.method=='L_RW_wo_origin'].reset_index()
        df = df[['patient', 'pred_day', 'deviation']]   

        # Percentage of predictions within clinically acceptable prediction error
        acceptable = df[(df.deviation > overprediction_limit) & (df.deviation < underprediction_limit)].reset_index(drop=True)
        perc_of_clinically_acceptable_pred = len(acceptable)/len(df)*100

        # Percentage of overpredictions
        overpred = df[df.deviation < overprediction_limit]
        perc_of_overpred = len(overpred)/len(df)*100

        # Percentage of underpredictions
        underpred = df[df.deviation > underprediction_limit]
        perc_of_underpred = len(underpred)/len(df)*100

        print(f'% of predictions within clinically acceptable prediction error: {perc_of_clinically_acceptable_pred:.2f}%, n = {len(df)}\n')
        print(f'% of predictions which are overpredictions: {perc_of_overpred:.2f}%, n = {len(df)}\n')
        print(f'% of predictions which are underpredictions: {perc_of_underpred:.2f}%, n = {len(df)}')

    sys.stdout = original_stdout

# Fig 4
def values_in_clinically_relevant_flow_chart(dose='total'):
    """
    Calculate values for clinically relevant flow chart, in the flow chart boxes, and in additional information

    Output: 
    - Printed values for flow chart boxes and additional information
    - Final dataframe with remaining predictions after all exclusions
    """
    original_stdout = sys.stdout

    file_string = 'clinically_relevant_flow_chart_' + dose + '.txt'
    with open(file_string, 'w') as f:
        sys.stdout = f

        if dose == 'total':
            result_file = 'CURATE_results.xlsx'
        else:
            result_file = result_file_evening

        df = create_df_for_CURATE_assessment(result_file)

        # 1. Calculate values for clinially relevant flow chart (step 1 of 2)

        total_predictions = len(df)
        unreliable = len(df[df.reliable==False])

        # Keep reliable predictions
        df = df[df.reliable==True].reset_index(drop=True)

        reliable = len(df)
        inaccurate = len(df[df.accurate==False])

        # Keep accurate predictions
        df = df[df.accurate==True].reset_index(drop=True)

        reliable_accurate = len(df)

        print(f'Flowchart numbers:\ntotal_predictions {total_predictions} | unreliable {unreliable} | reliable {reliable} | inaccurate {inaccurate} | reliable_accurate {reliable_accurate}')

        # 2. Calculate values for additional information

        reliable_accurate_actionable = len(df[df.actionable==True])

        # Keep actionable predictions
        df = df[df.actionable==True].reset_index(drop=True)

        reliable_accurate_actionable_diff_dose = len(df[df.diff_dose==True])

        # Keep diff dose predictions
        df = df[df.diff_dose==True].reset_index(drop=True)

        reliable_accurate_actionable_diff_dose_non_therapeutic_range = len(df[df.therapeutic_range==False])

        # Keep non therapeutic range
        df = df[df.therapeutic_range==False].reset_index(drop=True)

        print(f'\nAdditional information:\nreliable_accurate_actionable {reliable_accurate_actionable} out of {reliable_accurate} |\n\
        reliable_accurate_actionable_diff_dose {reliable_accurate_actionable_diff_dose} out of {reliable_accurate_actionable} |\n\
        reliable_accurate_actionable_diff_dose_non_therapeutic_range {reliable_accurate_actionable_diff_dose_non_therapeutic_range} out of {reliable_accurate_actionable_diff_dose}')

        try:
            # Add column for difference in doses recommended and administered
            df['diff_dose_recommended_and_administered'] = df['dose_recommendation'] - df['dose']
            df['diff_dose_recommended_and_administered'] = df['diff_dose_recommended_and_administered'].astype(float)
            result_and_distribution(df.diff_dose_recommended_and_administered, 'Dose recommended minus administered')
        except:
            print('dose difference calculation failed')
        

    sys.stdout = original_stdout

    return df

# Fig 5
def fig_5a_case_reach_TR_earlier(plot=False, all_data_file=all_data_file_total):
    """
    Scatter plot of response vs day for patient 120,
    with green marker for first day in therapeutic range, 
    and purple marker for potential first day with
    CURATE.AI. 
    """
    
    df, df_original = fig_5b_case_reach_TR_earlier()

    # Find predicted response on day 4
    predicted_response = (df_original.loc[0, 'coeff_1x'] * 2) + (df_original.loc[0, 'coeff_0x'])

    # SOC data
    patient_120 = pd.read_excel(all_data_file)
    patient_120 = patient_120[patient_120.patient==120]
    patient_120 = patient_120[['day', 'response']].reset_index(drop=True)

    if plot==True:
        # Plot
        fig, axes = plt.subplots(figsize=(7,7))
        sns.set(style='white', font_scale=2.2,
            rc={"xtick.bottom":True, "ytick.left":True})

        plt.plot(patient_120.day, patient_120.response, 'yo', linestyle='-', ms=10)
        plt.scatter(x=patient_120.day[0], y=patient_120.response[0], color='y', s=100, label='SOC dosing')
        plt.plot(4, predicted_response, 'm^', ms=10, label='First day of therapeutic range\nwith CURATE.AI-assisted dosing')
        plt.plot(8, 9.9, 'go', ms=10, label='First day of therapeutic range\nwith SOC dosing')

        plt.ylim(0,max(patient_120.response+1))

        sns.despine()
        plt.xticks(np.arange(2,max(patient_120.day),step=4))
        plt.xlabel('Day')
        plt.ylabel('TTL (ng/ml)')
        plt.axhspan(8, 10, facecolor='grey', alpha=0.2)

        handles, labels = plt.gca().get_legend_handles_labels()
        order = [2,1,0]
        legend1 = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                bbox_to_anchor=(1.04,0.5), loc='center left', frameon=False) 

        legend_elements = [Patch(facecolor='grey', edgecolor='grey',
                                label='Region within therapeutic range', alpha=.2)]
        legend2 = plt.legend(handles=legend_elements, bbox_to_anchor=(1.04,0.34), loc='upper left', frameon=False)

        axes.add_artist(legend1)
        axes.add_artist(legend2)
        
        plt.savefig('fig_5b_case_reach_TR_earlier.png', dpi=1000, facecolor='w', bbox_inches='tight')

    return patient_120

def fig_5b_case_reach_TR_earlier(plot=False, result_file=result_file_total):
    """
    Line plot of response vs dose for patient 120's day recommendation,
    with data points as (dose, response) pairs on day 2 and 3,
    and with linear regression line. 
    """
    df = pd.read_excel(result_file, sheet_name='result')

    # Subset patient 120 and method
    df = df[(df.patient==120) & (df.method=='L_RW_wo_origin') & (df.pred_day==4)]

    df = df[['patient', 'pred_day', 'fit_dose_1', 'fit_dose_2', 'fit_response_1', 'fit_response_2', 
             'coeff_1x', 'coeff_0x', 'dose', 'response', 'prediction', 'deviation', 'abs_deviation']].reset_index(drop=True)

    # Add dose recommendation columns for tac levels of 8 to 10 ng/ml
    df['dose_recommendation_8'] = ""
    df['dose_recommendation_10'] = ""

    for i in range(len(df)):

        # Create function
        coeff = df.loc[i, 'coeff_1x':'coeff_0x'].apply(float).to_numpy()
        coeff = coeff[~np.isnan(coeff)]
        p = np.poly1d(coeff)
        x = np.linspace(0, max_dose_recommendation)
        y = p(x)

        # Check for duplicates, which will occur if coeff_1x is very close to 0, and
        # will cause RuntimeError for interp1d. Hence, set interpolated doses to the intercept,
        # also known as coeff_0x
        dupes = [x for n, x in enumerate(y) if x in y[:n]]
        if len(dupes) != 0:
            df.loc[i, 'dose_recommendation_8'] = df.loc[i, 'coeff_0x']
            df.loc[i, 'dose_recommendation_10'] = df.loc[i, 'coeff_0x']

        else:
            f = interpolate.interp1d(y, x, fill_value='extrapolate')

            df.loc[i, 'dose_recommendation_8'] = f(8)
            df.loc[i, 'dose_recommendation_10'] = f(10)

    df_original = df.copy()

    df = df[['patient', 'pred_day', 'fit_dose_1', 'fit_dose_2', 'fit_response_1', 'fit_response_2']]

    # Create dataframe for x as doses to fit regression model
    df_1 = df[['patient', 'pred_day', 'fit_dose_1', 'fit_dose_2']]
    df_1 = df_1.set_index(['patient', 'pred_day'])
    df_1 = df_1.stack().reset_index()
    df_1 = df_1.rename(columns={'level_2':'day', 0:'x'})
    df_1['day'] = df_1['day'].replace({'fit_dose_1':2, 'fit_dose_2':3})

    # Create dataframe for y as response to fit regression model
    df_2 = df[['patient', 'pred_day', 'fit_response_1', 'fit_response_2']]
    df_2 = df_2.set_index(['patient', 'pred_day'])
    df_2 = df_2.stack().reset_index()
    df_2 = df_2.rename(columns={'level_2':'day', 0:'y'})
    df_2['day'] = df_2['day'].replace({'fit_response_1':2, 'fit_response_2':3})

    combined_df = df_1.merge(df_2, how='left', on=['patient', 'pred_day', 'day'])

    if plot == True:
        # Plot
        sns.set(style='white', font_scale=2.2,
            rc={"figure.figsize":(7,7), "xtick.bottom":True, "ytick.left":True})

        # Plot regression line
        x = np.array([combined_df.x[0],combined_df.x[1]])
        y = np.array([combined_df.y[0],combined_df.y[1]])
        a, b = np.polyfit(x, y, 1)
        x_values = np.linspace(0, 3)
        plt.plot(x_values, a*x_values + b, linestyle='-', color='y')

        # Plot scatter points
        plt.scatter(x, y, s=120, color='y')

        # Plot therapeutic range
        plt.axhspan(8, 10, facecolor='grey', alpha=0.2)

        # Label days
        for i in range(combined_df.shape[0]):
            plt.text(x=combined_df.x[i]+0.13,y=combined_df.y[i]-0.3,s=int(combined_df.day[i]),
                    fontdict=dict(color='black',size=18),
                    bbox=dict(facecolor='y', ec='black', alpha=0.5, boxstyle='circle'))

        sns.despine()
        plt.title('Day 4 Recommendation')
        plt.xlabel('Dose (mg)')
        plt.ylabel('TTL (ng/ml)')
        plt.xticks(np.arange(0,3.5,step=0.5))
        plt.xlim(0,2.5)
        
        plt.savefig('fig_5a_case_reach_TR_earlier.png', dpi=1000, facecolor='w', bbox_inches='tight')

    return combined_df, df_original

# Fig 6
def fig_6a_case_sustain_TR_longer(plot=False, dose='total'):
    """
    Multiple plots of response vs dose for repeated dosing strategy of 
    patient 118, with each plot representing one day of prediction. 
    """
    if dose == 'total':
        result_file = result_file_total
    else:
        result_file = result_file_evening

    dat_original, combined_df = fig_6_computation(plot, dose, result_file)

    # Subset repeated doses
    combined_df = combined_df[(combined_df.pred_day > 4) & (combined_df.pred_day <= 9)].reset_index(drop=True)

    sns.set(style='white', font_scale=2.2,
        rc={"xtick.bottom":True, "ytick.left":True})

    fig, ax = plt.subplots(1, 5, figsize=(25,5), gridspec_kw = {'wspace':0.3, 'hspace':0.5})
    # plt.subplots_adjust(wspace=-0.1)

    # Loop through number of predictions chosen
    for i in range(5):

        plt.subplot(1,5,i+1)

        # Plot regression lisne
        x = np.array([combined_df.x[i*2],combined_df.x[i*2+1]])
        y = np.array([combined_df.y[i*2],combined_df.y[i*2+1]])
        a, b = np.polyfit(x, y, 1)
        x_values = np.linspace(0, 9)
        plt.plot(x_values, a*x_values + b, linestyle='-', color='y')

        # Plot scatter points
        plt.scatter(x, y, s=100, color='y')

        plt.axhspan(8, 10, facecolor='grey', alpha=0.2)

        sns.despine()
        plt.ylim(0,max(combined_df.y+2))
        if i ==0:
            plt.ylabel('TTL (ng/ml)')
        plt.yticks(np.arange(0,15,step=2))
        plt.xlabel('Dose (mg)')
        plt.xticks(np.arange(0,9,step=2))
        plt.title('Day ' + str(combined_df.pred_day[i*2+1]) + '\nRecommendation')

        # Label days
        for j in range(2):
            plt.text(x=combined_df.x[i*2+j]+0.8,y=combined_df.y[i*2+j]-0.3,s=int(combined_df.day[i*2+j]), 
                fontdict=dict(color='black',size=18),
                bbox=dict(facecolor='y', ec='black', alpha=0.5, boxstyle='circle'))
            
        # Add legend for grey patch of therapeutic range
        if i == 0:
            legend_elements = [Patch(facecolor='grey', edgecolor='grey',
                                    label='Region within therapeutic range', alpha=.2)]
            plt.legend(handles=legend_elements, bbox_to_anchor=(-0.2,-.3), loc='upper left', frameon=False)       

    plt.tight_layout()
    plt.savefig('fig_6a_case_sustain_TR_longer' + dose + '.png',dpi=1000, bbox_inches='tight', pad_inches=0)

    return combined_df

def fig_6b_case_sustain_TR_longer(plot=False, dose='total'):
    """Line and scatter plot of repeated dose vs day for CURATE.AI-assisted vs SOC"""
    if dose == 'total':
        result_file = result_file_total
    else:
        result_file = result_file_evening

    dat_original, combined_df = fig_6_computation(plot, dose, result_file)
    clean_dat = pd.read_excel(result_file, sheet_name='clean')
    
    # Subset pred_days with repeated dose of 6mg
    dat = dat_original[(dat_original.pred_day >= 5) & (dat_original.pred_day <= 9)]
    CURATE_dosing = [7.5, 5.5, 5.5, 5, 5]
    dat['CURATE.AI-assisted dosing'] = CURATE_dosing

    # Subset columns
    dat = dat[['pred_day','CURATE.AI-assisted dosing']]
    dat = dat.rename(columns={'pred_day':'day'})

    # Subset patient 118 data only
    clean_dat = clean_dat[(clean_dat.patient == 118) & ((clean_dat.day >= 5) & (clean_dat.day <= 9))].reset_index(drop=True)

    # Subset day and dose
    clean_dat = clean_dat[['day', 'dose']]
    clean_dat = clean_dat.rename(columns={'dose':'SOC dosing'})

    # Combine both CURATE.AI-assisted dosing recommendations and actual dose given
    combined_dat = dat.merge(clean_dat, how='left', on='day')

    # Plot
    sns.set(font_scale=2.2, rc={"figure.figsize": (9,7), "xtick.bottom":True, "ytick.left":True}, style='white')

    plt.plot(combined_dat['day'], combined_dat['CURATE.AI-assisted dosing'], marker='^', color='m', label='CURATE.AI-assisted dosing', ms=10)
    plt.plot(combined_dat['day'], combined_dat['SOC dosing'], marker='o', color='y', label='SOC dosing', ms=10)
    plt.legend(bbox_to_anchor=(0.5,-0.5), loc='center', frameon=False)
    sns.despine()
    plt.xlabel('Day')
    plt.ylabel('Dose (mg)')
    plt.yticks(np.arange(5,8,step=0.5))
    plt.ylim(4.5, 8)
    

    plt.tight_layout()
    plt.savefig('fig_6b_case_sustain_TR_longer'+ dose +'.png',dpi=1000)

def fig_6c_case_sustain_TR_longer(result_file=result_file_total, plot=True, dose='total'):
    """Scatter plot of dose and response for CURATE.AI-assisted and SOC dosing"""
    dat_original, combined_df = fig_6_computation(plot=plot, dose=dose, result_file=result_file)
    clean_dat = pd.read_excel(result_file, sheet_name='clean')

    # Subset pred_days with repeated dose of 6mg
    dat = dat_original[(dat_original.pred_day >= 5) & (dat_original.pred_day <= 9)].reset_index(drop=True)

    # Add column for CURATE recommendation
    CURATE_dosing = [7.5, 5.5, 5.5, 5, 5]
    dat['CURATE-recommended dose'] = CURATE_dosing

    # Add column for predicted response if CURATE dose was administered instead
    dat['predicted_response_based_on_rec'] = dat['coeff_1x'] * dat['CURATE-recommended dose'] + dat['coeff_0x']

    dat = dat[['pred_day', 'dose', 'response', 'CURATE-recommended dose', 'predicted_response_based_on_rec']]

    # Plot
    fig, axes = plt.subplots()
    sns.set(font_scale=2.2, rc={"figure.figsize": (7,7), "xtick.bottom":True, "ytick.left":True}, style='white')

    plt.scatter(x=dat['CURATE-recommended dose'], y=dat['predicted_response_based_on_rec'], marker='^', color='m', label='CURATE.AI-assisted dosing', s=100)
    plt.scatter(x=dat['dose'], y=dat['response'], marker='o', color='y', label='SOC dosing', s=100)
    sns.despine()
    plt.xlabel('Dose (mg)')
    plt.ylabel('TTL (ng/ml)')
    plt.axhspan(therapeutic_range_lower_limit, therapeutic_range_upper_limit, facecolor='grey', alpha=0.2)
    plt.xticks(np.arange(4,8.5,step=0.5))
    plt.xlim(4,8)
    plt.yticks(np.arange(8,15,step=1))

    legend1 = plt.legend(bbox_to_anchor=(0.5,-0.3), loc='center', frameon=False)

    legend_elements = [Patch(facecolor='grey', edgecolor='grey',
                            label='Region within therapeutic range', alpha=.2)]
    legend2 = plt.legend(handles=legend_elements, bbox_to_anchor=(-0.07,-0.35), loc='upper left', frameon=False)

    
    axes.add_artist(legend1)
    axes.add_artist(legend2)        

    for i in range(dat.shape[0]):
        plt.text(x=dat.dose[i]+0.2,y=dat.response[i],s=int(dat.pred_day[i]),
                 fontdict=dict(color='black',size=13),
                 bbox=dict(facecolor='y', ec='black', alpha=0.5, boxstyle='circle'))

        plt.text(x=dat.loc[i, 'CURATE-recommended dose']+0.2,y=dat.loc[i, 'predicted_response_based_on_rec'],s=int(dat.pred_day[i]),
             fontdict=dict(color='black',size=13),
             bbox=dict(facecolor='m', ec='black', alpha=0.5, boxstyle='circle'))

    plt.tight_layout()
    plt.savefig('fig_6c_case_sustain_TR_longer.png',dpi=1000, bbox_inches='tight')
    
    return dat

def fig_6_computation(plot, dose, result_file):
    """
    Plot RW profiles for patient 118, with shaded region representing therapeutic range,
    colors representing prediction days, and number circles for the day from which
    the dose-response pairs were obtained from.
    """    
    dat = pd.read_excel(result_file, sheet_name='result')
    # Subset L_RW_wo_origin and patient 118
    dat = dat[(dat.method=='L_RW_wo_origin') &  (dat.patient==118)]

    dat = dat[['patient', 'method', 'pred_day', 'dose', 'response', 'coeff_1x', 'coeff_0x', 'prediction', 'deviation', 'fit_dose_1', 'fit_dose_2', 'fit_response_1', 'fit_response_2', 'day_1', 'day_2']].reset_index(drop=True)

    # Interpolate to find percentage of possible dosing events for when prediction and observed response are outside range
    for i in range(len(dat)):
        # Create function
        coeff = dat.loc[i, 'coeff_1x':'coeff_0x'].apply(float).to_numpy()
        coeff = coeff[~np.isnan(coeff)]
        p = np.poly1d(coeff)
        x = np.linspace(0, max_dose_recommendation)
        y = p(x)

        # Check for duplicates, which will occur if coeff_1x is very close to 0, and
        # will cause RuntimeError for interp1d. Hence, set interpolated doses to the intercept,
        # also known as coeff_0x
        dupes = [x for n, x in enumerate(y) if x in y[:n]]
        if len(dupes) != 0:
            dat.loc[i, 'interpolated_dose_8'] = dat.loc[i, 'coeff_0x']
            dat.loc[i, 'interpolated_dose_9'] = dat.loc[i, 'coeff_0x']
            dat.loc[i, 'interpolated_dose_10'] = dat.loc[i, 'coeff_0x']

        else:
            f = interpolate.interp1d(y, x, fill_value='extrapolate')

            dat.loc[i, 'interpolated_dose_8'] = f(8)
            dat.loc[i, 'interpolated_dose_9'] = f(9)
            dat.loc[i, 'interpolated_dose_10'] = f(10)
        
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

    dat_original = dat.copy()

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
    
    # if plot==True:
    #     # Plot
    #     sns.set(font_scale=1.2, rc={"figure.figsize": (16,10), "xtick.bottom":True, "ytick.left":True},
    #             style='white')
    #     g = sns.lmplot(data=combined_df, x='x', y='y', hue='pred_day', ci=None, legend=False)

    #     ec = colors.to_rgba('black')
    #     ec = ec[:-1] + (0.3,)

    #     for i in range(combined_df.shape[0]):
    #         plt.text(x=combined_df.x[i]+0.3,y=combined_df.y[i]+0.3,s=int(combined_df.day[i]), 
    #         fontdict=dict(color='black',size=13),
    #         bbox=dict(facecolor='white', ec='black', alpha=0.5, boxstyle='circle'))
            
    #         plt.text(x=0+0.3,y=10.4+0.3,s=12, 
    #         fontdict=dict(color='black',size=13),
    #         bbox=dict(facecolor='white', ec=ec, boxstyle='circle'))
            
    #         plt.text(x=0+0.3,y=8.7+0.3,s=14, 
    #         fontdict=dict(color='black',size=13),
    #         bbox=dict(facecolor='white', ec=ec, boxstyle='circle'))

    #     plt.legend(bbox_to_anchor=(1.06,0.5), loc='center left', title='Day of Prediction', frameon=False)
    #     plt.xlabel('Tacrolimus dose (mg)')
    #     plt.ylabel('Tacrolimus level (ng/ml)')
    #     plt.axhspan(8, 10, facecolor='grey', alpha=0.2)

    #     # Add data point of day 12 and day 14
    #     plt.plot(0, 10.4, marker="o", markeredgecolor="black", markerfacecolor="white")
    #     plt.plot(0, 8.7, marker="o", markeredgecolor="black", markerfacecolor="white")
    #     plt.savefig('patient_118_RW_profiles_' + dose + '.png', dpi=500, facecolor='w', bbox_inches='tight')

    return dat_original, combined_df

# Effect of CURATE.AI
def effect_of_CURATE_categories(dose='total'):
    """
    Calculate and print out the percentage of days with projected improvement, 
    worsening, or having no effect on patient outcomes.
    """
    df = effect_of_CURATE(dose=dose)
    df = df.dropna().reset_index(drop=True)

    for i in range(len(df)):
        if 'Unaffected' in df['Effect of CURATE.AI-assisted dosing'][i]:
            df.loc[i, 'result'] = 'unaffected'
        elif 'Improve' in df['Effect of CURATE.AI-assisted dosing'][i]:
            df.loc[i, 'result'] = 'improve'
        elif 'Worsen' in df['Effect of CURATE.AI-assisted dosing'][i]:
            df.loc[i, 'result'] = 'worsen'
        else: print(f'uncertain result at index {i}')
        
    perc_of_days_improved = len(df[df.result=='improve'])/len(df)*100
    perc_of_days_worsened = len(df[df.result=='worsen'])/len(df)*100
    perc_of_days_unaffected = len(df[df.result=='unaffected'])/len(df)*100

    original_stdout = sys.stdout
    with open('effect_of_CURATE_categories_' + dose + '.txt', 'w') as f:
        sys.stdout = f
        print(f'perc_of_days_improved: {perc_of_days_improved:.2f}%, n = {len(df)}')
        print(f'perc_of_days_worsened: {perc_of_days_worsened:.2f}%, n = {len(df)}')
        print(f'perc_of_days_unaffected: {perc_of_days_unaffected:.2f}%, n = {len(df)}')
    sys.stdout = original_stdout

    return df

def effect_of_CURATE_values(dose='total'):
    """
    Output: 
    1) Print:
    - 1. % of days within therapeutic range
    - 2. % of participants that reach within first week
    - 3. Day where patient first achieved therapeutic range
    2) Corresponding dataframes
    """
    original_stdout = sys.stdout
    with open('effect_of_CURATE_' + dose + '.txt', 'w') as f:
        sys.stdout = f

        df = effect_of_CURATE(dose=dose)

        # Drop rows where response is NaN
        df = df[df.response.notna()].reset_index(drop=True)

        # Create column of final therapeutic range result
        for i in range(len(df)):
            if 'non' in df['Effect of CURATE.AI-assisted dosing'][i]:
                df.loc[i, 'final_response_in_TR'] = False
            else:
                df.loc[i, 'final_response_in_TR'] = True

        # 1a. % of days within therapeutic range
        perc_days_within_TR = df.groupby('patient')['final_response_in_TR'].apply(lambda x: x.sum()/x.count()*100)
        perc_days_within_TR = perc_days_within_TR.reset_index(name='result')
        result_and_distribution(perc_days_within_TR.result, '1a. % of days within therapeutic range (CURATE)')


        # 1b. % of days within therapeutic range in SOC
        # Drop rows where response is NaN
        data = fig_2_TTL_over_time(plot=False)
        data = data[data.response.notna()].reset_index(drop=True)

        # Add therapeutic range column
        for i in range(len(data)):
            if (data.response[i] >= therapeutic_range_lower_limit) & (data.response[i] <= therapeutic_range_upper_limit):
                data.loc[i, 'therapeutic_range'] = True
            else:
                data.loc[i, 'therapeutic_range'] = False

        perc_therapeutic_range = data.groupby('patient')['therapeutic_range'].apply(lambda x: x.sum()/x.count()*100)
        perc_therapeutic_range = perc_therapeutic_range.to_frame().reset_index()

        # Result and distribution
        result_and_distribution(perc_therapeutic_range.therapeutic_range, '1b. % of days within therapeutic range (SOC)')


        # 1c. Comparison between 1a and 1b
        if (stats.shapiro(perc_days_within_TR.result).pvalue > 0.05) & (stats.shapiro(perc_therapeutic_range.therapeutic_range).pvalue > 0.05):
            print(f'Normal, paired, paired t-test p-value: {stats.ttest_rel(perc_days_within_TR.result, perc_therapeutic_range.therapeutic_range).pvalue:.2f}')
        else:
            print(f'Non-normal, paired, wilcoxon signed-rank test p-value: {wilcoxon(perc_days_within_TR.result, perc_therapeutic_range.therapeutic_range).pvalue:.2f}\n')

        # 2. % of participants that reach within first week
        reach_TR_in_first_week = df[df.final_response_in_TR==True].groupby('patient')['day'].first().reset_index(name='first_day')
        reach_TR_in_first_week['result'] = reach_TR_in_first_week['first_day'] <= 7
        result = reach_TR_in_first_week['result'].sum() / len(df.patient.unique()) * 100
        print(f'2. % of participants that reach within first week: {result:.2f}, {reach_TR_in_first_week["result"].sum()} out of {len(df.patient.unique())} patients\n')

        # 3a. Day where patient first achieved therapeutic range in CURATE
        result_and_distribution(reach_TR_in_first_week.first_day, '3a. Day where patient first achieved therapeutic range (CURATE)')

        # 3b. Day where patient first achieved therapeutic range in SOC
        first_TR_df = data.copy()
        first_TR_df = first_TR_df[first_TR_df['Tacrolimus trough levels (TTL)']=='Therapeutic range'].reset_index(drop=True)
        first_TR_df = first_TR_df.groupby('patient')['Day'].first().to_frame().reset_index()

        # Result and distribution
        result_and_distribution(first_TR_df.Day, '3b. Day where patient first achieved therapeutic range (SOC)')

        # 3c. Compare between 3a and 3b
        if (stats.shapiro(reach_TR_in_first_week.first_day).pvalue > 0.05) & (stats.shapiro(first_TR_df.Day).pvalue > 0.05):
            print(f'Normal, paired, paired t-test p-value: {stats.ttest_rel(reach_TR_in_first_week.first_day, first_TR_df.Day).pvalue:.2f}')
        else:
            print(f'Non-normal, paired, wilcoxon signed-rank test p-value: {wilcoxon(reach_TR_in_first_week.first_day, first_TR_df.Day).pvalue:.2f}')

    sys.stdout = original_stdout

def effect_of_CURATE_inter_indiv_differences(dose='total'):
    """
    1) Print the percentage of patients, out of total patients, 
    that were in therapeutic range more/less/equally frequent with CURATE.AI-assisted
    dosing. 
    2) Print the percentage of patients, out of total patients, 
    that first achieved the therapeutic range earlier/later/on the same day
    with CURATE.AI-assisted dosing.
    """
    original_stdout = sys.stdout
    with open('effect_of_CURATE_inter_indiv_differences_' + dose + '.txt', 'w') as f:
        sys.stdout = f
        df = effect_of_CURATE().dropna()

        # Percentage of days within TTR
        SOC_TTR = df.groupby('patient')['therapeutic_range'].apply(lambda x: (x=='therapeutic').sum()/x.count()*100).reset_index(name='SOC_TTR')
        CURATE_TTR = df.groupby('patient')['Effect of CURATE.AI-assisted dosing'].apply(lambda x: (x.count()-x[x.str.contains('non-therapeutic')].count())/x.count()*100).reset_index(name='CURATE_TTR')

        combined_df_TTR = SOC_TTR.merge(CURATE_TTR, on='patient')
        for i in range(len(combined_df_TTR)):
            if combined_df_TTR.CURATE_TTR[i] > combined_df_TTR.SOC_TTR[i]:
                combined_df_TTR.loc[i, 'effect_of_CURATE_on_TTR'] = 'more frequent'
            elif combined_df_TTR.CURATE_TTR[i] < combined_df_TTR.SOC_TTR[i]:
                combined_df_TTR.loc[i, 'effect_of_CURATE_on_TTR'] = 'less frequent'
            else:
                combined_df_TTR.loc[i, 'effect_of_CURATE_on_TTR'] = 'equally frequent'

        TTR = combined_df_TTR.effect_of_CURATE_on_TTR.value_counts()/len(df.patient.unique())*100
        print(TTR)
        print(f'Out of {len(df.patient.unique())} patients')

        # First day to reach TTR
        SOC_first_day = df[df.therapeutic_range=='therapeutic'].groupby('patient')['day'].first().reset_index(name='SOC_first_day')
        CURATE_first_day = df.copy()
        CURATE_first_day = CURATE_first_day[CURATE_first_day['Effect of CURATE.AI-assisted dosing'].str.contains("non-therapeutic")==False].groupby('patient')['day'].first().reset_index(name='CURATE_first_day')

        combined_df_first_day = CURATE_first_day.merge(SOC_first_day, on='patient')
        for i in range(len(combined_df_first_day)):
            if combined_df_first_day.CURATE_first_day[i] < combined_df_first_day.SOC_first_day[i]:
                combined_df_first_day.loc[i, 'first_day_in_TTR'] = 'earlier'
            elif combined_df_first_day.CURATE_first_day[i] > combined_df_first_day.SOC_first_day[i]:
                combined_df_first_day.loc[i, 'first_day_in_TTR'] = 'later'
            else:
                combined_df_first_day.loc[i, 'first_day_in_TTR'] = 'same'

        first_day_in_TTR = combined_df_first_day.first_day_in_TTR.value_counts()/len(df.patient.unique())*100
        print(first_day_in_TTR)
        print(f'Out of {len(df.patient.unique())} patients')
    
    sys.stdout = original_stdout

# Fig 7a
def effect_of_CURATE(plot=False, dose='total'):
    """
    Facetgrid scatter plot of effect of CURATE on all data.

    Output:
    - Plot (saved)
    - Dataframe used to create the plot
    """
    if dose == 'total':
        all_data_file = all_data_file_total
    else:
        all_data_file = all_data_file_evening

    df = create_df_for_CURATE_assessment(dose=dose)

    # Add column of 'Effect of CURATE.AI-assisted dosing' and 
    # add column for dose range
    df['Effect of CURATE.AI-assisted dosing'] = ""

    for i in range(len(df)):
        if (df.effect_of_CURATE[i] == 'Unaffected') & (df.therapeutic_range[i] == True):
            df.loc[i, 'Effect of CURATE.AI-assisted dosing'] = 'Unaffected, remain as therapeutic range'
        elif (df.effect_of_CURATE[i] == 'Unaffected') & (df.therapeutic_range[i] == False):
            df.loc[i, 'Effect of CURATE.AI-assisted dosing'] = 'Unaffected, remain as non-therapeutic range'
        elif df.effect_of_CURATE[i] == 'Improve':
            df.loc[i, 'Effect of CURATE.AI-assisted dosing'] = 'Improve to therapeutic range'
        else:
            df.loc[i, 'Effect of CURATE.AI-assisted dosing'] = 'Worsen to non-therapeutic range'

    # Rename column in df from 'pred_day' to 'day' before merging
    df = df.rename(columns={'pred_day':'day'})

    # Subset columns
    df = df[['patient', 'day', 'Effect of CURATE.AI-assisted dosing']]

    # Import all data
    all_data = pd.read_excel(all_data_file)

    # Subset patient, day, response, dose 
    all_data = all_data[['patient', 'day', 'response', 'dose']]

    # Add therapeutic range column
    all_data['therapeutic_range'] = ""
    for i in range(len(all_data)):
        if (all_data.response[i] >= therapeutic_range_lower_limit) & (all_data.response[i] <= therapeutic_range_upper_limit):
            all_data.loc[i, 'therapeutic_range'] = 'therapeutic'
        else:
            all_data.loc[i, 'therapeutic_range'] = 'non_therapeutic'

    # Merge on all columns in all_data
    combined_dat = all_data.merge(df, how='left', on=['patient', 'day'])

    # Add dose range column and fill in 'Effect of CURATE'
    combined_dat['Dose range'] = ""

    for i in range(len(combined_dat)):
        if np.isnan(combined_dat.dose[i]):
            combined_dat.loc[i, 'Dose range'] = 'Unavailable'
        elif combined_dat.dose[i] < low_dose_upper_limit:
            combined_dat.loc[i, 'Dose range'] = 'Low'
        elif combined_dat.dose[i] < medium_dose_upper_limit:
            combined_dat.loc[i, 'Dose range'] = 'Medium'
        else: 
            combined_dat.loc[i, 'Dose range'] = 'High'

        if str(combined_dat['Effect of CURATE.AI-assisted dosing'][i])=='nan':
            if combined_dat.therapeutic_range[i] == 'therapeutic':
                combined_dat.loc[i, 'Effect of CURATE.AI-assisted dosing'] = 'Unaffected, remain as therapeutic range'
            else:
                combined_dat.loc[i, 'Effect of CURATE.AI-assisted dosing'] = 'Unaffected, remain as non-therapeutic range' 

    # Rename patients
    combined_dat['patient'] = combined_dat['patient'].map({84:1, 114:2, 117:3, 118:4, 120:5, 121:6, 122:7,
                                                123:8, 125:9, 126:10, 129:11, 130:12, 131:13, 132:14,
                                                133:15, 138:16})

    if plot == True:
        # Plot
        sns.set(font_scale=1.5, rc={"figure.figsize": (16,10), "xtick.bottom":True, "ytick.left":True}, style='white')
        hue_order = ['Unaffected, remain as therapeutic range', 'Unaffected, remain as non-therapeutic range',
                    'Improve to therapeutic range', 'Worsen to non-therapeutic range']
        palette = [sns.color_palette()[1], sns.color_palette()[0], sns.color_palette()[2],\
                sns.color_palette()[3]]
        style_order = ['Low', 'Medium', 'High', 'Unavailable']

        # Scatter point
        g = sns.relplot(data=combined_dat, x='day', y='response', hue='Effect of CURATE.AI-assisted dosing',\
                        hue_order=hue_order, col='patient', palette=palette,\
                        col_wrap=4, height=3, aspect=1, s=100, style_order=style_order, zorder=2)

        # Move legend below plot
        sns.move_legend(g, 'center', bbox_to_anchor=(0.20,-0.1), title='Effect of CURATE.AI-assisted dosing', ncol=1)

        # Titles and labels
        g.set_titles('Patient {col_name}')
        g.set(yticks=np.arange(0,math.ceil(max(combined_dat['response'])),4),
            xticks=np.arange(0,max(combined_dat.day),step=5))
        g.set_ylabels('TTL (ng/ml)')
        g.set_xlabels('Day')

        # Add gray region for therapeutic range
        for ax in g.axes:
            ax.axhspan(therapeutic_range_lower_limit, therapeutic_range_upper_limit, facecolor='grey', alpha=0.2, zorder=1)

        legend1 = plt.legend()
        legend_elements = [Patch(facecolor='grey', edgecolor='grey',
                            label='Region within\ntherapeutic range', alpha=.2)]
        legend2 = plt.legend(handles=legend_elements, bbox_to_anchor=(-1.1,-0.5), loc='upper left', frameon=False)

        # plt.show()
        # plt.tight_layout()
        plt.savefig('effect_of_CURATE_'+dose+'.png', dpi=1000, facecolor='w', bbox_inches='tight')

    return combined_dat

# Fig 7b
def SOC_CURATE_perc_in_TR(dose='total'):
    """
    Boxplot of % of days in TR, for SOC and CURATE.
    Print out kruskal wallis test for difference in medians.
    """

    # SOC
    perc_days_TR_SOC = fig_2_TTL_over_time(plot=False, dose=dose)

    # Drop rows where response is NaN
    perc_days_TR_SOC = perc_days_TR_SOC[perc_days_TR_SOC.response.notna()].reset_index(drop=True)

    # Add therapeutic range column
    for i in range(len(perc_days_TR_SOC)):
        if (perc_days_TR_SOC.response[i] >= therapeutic_range_lower_limit) & (perc_days_TR_SOC.response[i] <= therapeutic_range_upper_limit):
            perc_days_TR_SOC.loc[i, 'therapeutic_range'] = True
        else:
            perc_days_TR_SOC.loc[i, 'therapeutic_range'] = False

    perc_days_TR_SOC = perc_days_TR_SOC.groupby('patient')['therapeutic_range'].apply(lambda x: x.sum()/x.count()*100)
    perc_days_TR_SOC = perc_days_TR_SOC.reset_index(name='SOC')

    # CURATE
    perc_days_TR_CURATE = effect_of_CURATE(dose=dose)

    # Drop rows where response is NaN
    perc_days_TR_CURATE = perc_days_TR_CURATE[perc_days_TR_CURATE.response.notna()].reset_index(drop=True)

    # Create column of final therapeutic range result
    for i in range(len(perc_days_TR_CURATE)):
        if 'non' in perc_days_TR_CURATE['Effect of CURATE.AI-assisted dosing'][i]:
            perc_days_TR_CURATE.loc[i, 'final_response_in_TR'] = False
        else:
            perc_days_TR_CURATE.loc[i, 'final_response_in_TR'] = True

    perc_days_TR_CURATE = perc_days_TR_CURATE.groupby('patient')['final_response_in_TR'].apply(lambda x: x.sum()/x.count()*100)
    perc_days_TR_CURATE = perc_days_TR_CURATE.reset_index(name='CURATE')

    perc_days_TR = perc_days_TR_SOC.merge(perc_days_TR_CURATE, how='left', on='patient')

    # Compare medians
    print('Comparison of medians between SOC and CURATE\n')
    if (stats.shapiro(perc_days_TR.SOC).pvalue < 0.05) or (stats.shapiro(perc_days_TR.CURATE).pvalue < 0.05):
        print(f'Non-normal distribution, Wilcoxon p value: {wilcoxon(perc_days_TR.SOC, perc_days_TR.CURATE).pvalue:.2f}')
    else:
        print(f'Normal distribution, Paired t test p value: {stats.ttest_rel(perc_days_TR.SOC, perc_days_TR.CURATE).pvalue:.2f}')

    # Rearrange dataframe for seaborn boxplot
    perc_days_TR_plot = perc_days_TR.set_index('patient')
    perc_days_TR_plot = perc_days_TR_plot.stack().reset_index()
    perc_days_TR_plot = perc_days_TR_plot.rename(columns={'level_1':'Dosing', 0:'Days in therapeutic range (%)'})
    
    return perc_days_TR

def plot_SOC_CURATE_perc_in_TR():
    df = SOC_CURATE_perc_in_TR()

    fig, ax = plt.subplots()
    # plt.figure(figsize=(5,5))
    sns.set(font_scale=1.2, rc={"xtick.bottom":True, "ytick.left":True}, style='white')

    # Bar plot
    p = ax.bar(['SOC dosing\n(N = 16)', 'CURATE.AI-assisted\ndosing\n(N = 16)'], [mean(df.SOC), mean(df.CURATE)], yerr=[stdev(df.SOC), stdev(df.CURATE)],
        edgecolor='black', capsize=10, color=[sns.color_palette("Paired",8)[0],sns.color_palette("Paired",8)[1]], zorder=1, width=.4)

    # Scatter points
    plt.scatter(np.zeros(len(df.SOC)), df.SOC, c='k', zorder=2)
    plt.scatter(np.ones(len(df.CURATE)), df.CURATE, c='k', zorder=3)
    for i in range(len(df.CURATE)):
        plt.plot([0,1], [df.SOC[i], df.CURATE[i]], c='k', alpha=.5)

    # Aesthetics
    plt.ylabel('Days in therapeutic range (%)')
    sns.despine()
    plt.ylim(0, 65)

    # Bar labels
    rects = ax.patches
    averages = [mean(df.SOC), mean(df.CURATE)]
    SD = [stdev(df.SOC), stdev(df.CURATE)]
    labels = [f"{i:.2f} ± {j:.2f}" for i,j in zip(averages, SD)]
    for rect, label in zip(rects, labels):
        height = rect.get_height()
        ax.text(
            rect.get_x() + rect.get_width() / 2, height + 35, label, 
            ha="center", va="bottom", fontsize=13
        )

    plt.tight_layout()
    plt.savefig('perc_days_in_TR.png', dpi=1000, facecolor='w', bbox_inches='tight')

    return df

# Fig 7c
def SOC_CURATE_first_day_in_TR(plot=False, dose='total'):
    """
    Boxplot for day when TR is first achieved, for
    both SOC and CURATE
    """

    # SOC
    SOC = fig_2_TTL_over_time(plot=False, dose=dose)
    SOC = SOC[SOC.response.notna()].reset_index(drop=True)

    # Add therapeutic range column
    for i in range(len(SOC)):
        if (SOC.response[i] >= therapeutic_range_lower_limit) & (SOC.response[i] <= therapeutic_range_upper_limit):
            SOC.loc[i, 'therapeutic_range'] = True
        else:
            SOC.loc[i, 'therapeutic_range'] = False
    print(SOC.columns)

    SOC = SOC[SOC['Tacrolimus trough levels (TTL)']=='Therapeutic range'].reset_index(drop=True)
    SOC = SOC.groupby('patient')['Day'].first().reset_index(name='SOC')

    # CURATE
    CURATE = effect_of_CURATE(dose=dose)

    # Drop rows where response is NaN
    CURATE = CURATE[CURATE.response.notna()].reset_index(drop=True)

    # Create column of final therapeutic range result
    for i in range(len(CURATE)):
        if 'non' in CURATE['Effect of CURATE.AI-assisted dosing'][i]:
            CURATE.loc[i, 'final_response_in_TR'] = False
        else:
            CURATE.loc[i, 'final_response_in_TR'] = True

    CURATE = CURATE[CURATE.final_response_in_TR==True].groupby('patient')['day'].first().reset_index(name='CURATE')

    # Merge SOC and CURATE into one dataframe
    combined_df = SOC.merge(CURATE, how='left', on='patient')

    # Compare medians
    print('Comparison of medians between SOC and CURATE\n')
    if (stats.shapiro(combined_df.SOC).pvalue < 0.05) or (stats.shapiro(combined_df.CURATE).pvalue < 0.05):
        print(f'Non-normal distribution, Wilcoxon p value: {wilcoxon(combined_df.SOC, combined_df.CURATE).pvalue:.2f}')
    else:
        print(f'Normal distribution, Paired t test p value: {stats.ttest_rel(combined_df.SOC, combined_df.CURATE).pvalue:.2f}')

    # Rearrange dataframe for seaborn boxplot
    plot_df = combined_df.set_index('patient')
    plot_df = plot_df.stack().reset_index()
    plot_df = plot_df.rename(columns={'level_1':'Dosing', 0:'First day in therapeutic range'})

    if plot == True:
        
        fig, ax = plt.subplots()
        sns.set(font_scale=1.2, rc={"figure.figsize": (5,5), "xtick.bottom":True, "ytick.left":True}, style='white')
        
        # Boxplot
        g = sns.boxplot(x="Dosing", y="First day in therapeutic range", 
                        data=plot_df, width=0.5, palette=[sns.color_palette("Paired",8)[2],sns.color_palette("Paired",8)[3]],
                        zorder=1)
        
        # Scatter points
        SOC_df = plot_df[plot_df.Dosing=='SOC'].reset_index(drop=True)['First day in therapeutic range']
        CURATE_df = plot_df[plot_df.Dosing=='CURATE'].reset_index(drop=True)['First day in therapeutic range']
        
        plt.scatter(np.zeros(len(SOC_df)), SOC_df, c='k', zorder=2)
        plt.scatter(np.ones(len(CURATE_df)), CURATE_df, c='k', zorder=3)
        for i in range(len(SOC_df)):
            plt.plot([0,1], [SOC_df[i], CURATE_df[i]], c='k', alpha=.5)

        # Aesthetics
        sns.despine()
        g.set_xlabel(None)
        # g.set_ylabel('Days in therapeutic range (%)')
        g.set_xticklabels(['SOC dosing\n(N = 15)', 'CURATE.AI-assisted\ndosing\n(N = 15)'])

        # Bracket and star
        x1, x2 = 0, 1
        y, h = SOC_df.max() + 3.5, 1
        plt.plot([0, 0, 1, 1], [y, y+h, y+h, y], lw=1.5, c='k')
        plt.text((x1+x2)*.5, y+h, "*", ha='center', va='bottom', color='k')    

        # Box labels
        rects = ax.patches
        medians = [median(SOC_df), median(CURATE_df)]
        lower_quartile = [SOC_df.quantile(0.25), CURATE_df.quantile(0.25)]
        upper_quartile = [SOC_df.quantile(0.75), CURATE_df.quantile(0.75)]
        labels = [f"{i:.2f}\n({j:.2f} - {k:.2f})" for i,j,k in zip(medians, lower_quartile, upper_quartile)]
        
        ax.text(0, SOC_df.max()+0.9, labels[0], ha='center', va='bottom', 
                color='k', fontsize=13)
        ax.text(1, CURATE_df.max()+1.5, labels[1], ha='center', va='bottom', 
                color='k', fontsize=13)

        # plt.show()
        # Save
        plt.savefig('SOC_CURATE_first_day_in_TR_'+dose+'.png', dpi=1000, facecolor='w', bbox_inches='tight')

    return plot_df, SOC_df, CURATE_df

# Fig 7d
def SOC_CURATE_perc_pts_TR_in_first_week(plot=False, dose='total'):
    """
    Barplot of % of patients in TR within first week, of
    SOC and CURATE.
    """
    # SOC
    data = fig_2_TTL_over_time(plot=False, dose=dose)

    # Drop rows where response is NaN
    data = data[data.response.notna()].reset_index(drop=True)

    # Add therapeutic range column
    for i in range(len(data)):
        if (data.response[i] >= therapeutic_range_lower_limit) & (data.response[i] <= therapeutic_range_upper_limit):
            data.loc[i, 'therapeutic_range'] = True
        else:
            data.loc[i, 'therapeutic_range'] = False

    first_week_df = data.copy()
    print(first_week_df.columns)
    first_week_df = first_week_df[first_week_df['Tacrolimus trough levels (TTL)']=='Therapeutic range'].reset_index(drop=True)
    first_week_df = (first_week_df.groupby('patient')['Day'].first() <= 7).to_frame().reset_index()
    result = first_week_df.Day.sum()/first_week_df.Day.count()*100

    SOC = result

    # CURATE
    df = effect_of_CURATE()
    # Drop rows where response is NaN
    df = df[df.response.notna()].reset_index(drop=True)

    # Create column of final therapeutic range result
    for i in range(len(df)):
        if 'non' in df['Effect of CURATE.AI-assisted dosing'][i]:
            df.loc[i, 'final_response_in_TR'] = False
        else:
            df.loc[i, 'final_response_in_TR'] = True

    # % of participants that reach within first week
    reach_TR_in_first_week = df[df.final_response_in_TR==True].groupby('patient')['day'].first().reset_index(name='first_day')
    reach_TR_in_first_week['result'] = reach_TR_in_first_week['first_day'] <= 7
    result = reach_TR_in_first_week['result'].sum() / len(reach_TR_in_first_week) * 100

    CURATE = result

    # Restructure dataframe for plotting with seaborn
    plot_df = pd.DataFrame({'SOC':[SOC], 'CURATE':[CURATE]}).stack()
    plot_df = plot_df.to_frame().reset_index()
    plot_df = plot_df.rename(columns={'level_1':'Dosing', 0:'perc_reach_TR_in_first_week'})

    # Plot
    if plot == True:
        sns.set(font_scale=1.2, rc={"figure.figsize": (5,5), "xtick.bottom":True, "ytick.left":True}, style='white')
        fig, ax = plt.subplots()
        p = ax.bar(plot_df.Dosing, plot_df.perc_reach_TR_in_first_week, width=.5, 
        color=[sns.color_palette("Paired",10)[8],sns.color_palette("Paired",10)[9]], edgecolor='k')
        
        # Aesthetics
        sns.despine()
        ax.set_xticklabels(['SOC dosing\n(N = 15)', 'CURATE.AI-assisted\ndosing\n(N = 15)'])
        plt.ylabel('Patients who achieve therapeutic\nrange in first week (%)')
        
        # Bar labels
        ax.bar_label(p, fmt='%.2f', fontsize=13)
        # plt.show()
        plt.savefig('SOC_CURATE_perc_pts_TR_in_first_week_'+dose+'.png', dpi=1000, facecolor='w', bbox_inches='tight')

    return plot_df

# Assessment of CURATE.AI
def create_df_for_CURATE_assessment(result_file = result_file_total, dose='total'):
    if dose == 'total':
        result_file = result_file_total
    else:
        result_file = result_file_evening
    
    # Import output results
    dat = pd.read_excel(result_file, sheet_name='result')
    # dat_dose_by_mg = pd.read_excel(result_file, sheet_name='clean')

    # Subset L_RW_wo_origin
    dat = dat[dat.method=='L_RW_wo_origin'].reset_index(drop=True)

    # Add dose recommendation columns for tac levels of 8 to 10 ng/ml
    dat['dose_recommendation_8'] = ""
    dat['dose_recommendation_10'] = ""

    for i in range(len(dat)):

        # Create function
        coeff = dat.loc[i, 'coeff_1x':'coeff_0x'].apply(float).to_numpy()
        coeff = coeff[~np.isnan(coeff)]
        p = np.poly1d(coeff)
        x = np.linspace(0, max_dose_recommendation)
        y = p(x)

        # Check for duplicates, which will occur if coeff_1x is very close to 0, and
        # will cause RuntimeError for interp1d. Hence, set interpolated doses to the intercept,
        # also known as coeff_0x
        dupes = [x for n, x in enumerate(y) if x in y[:n]]
        if len(dupes) != 0:
            dat.loc[i, 'dose_recommendation_8'] = dat.loc[i, 'coeff_0x']
            dat.loc[i, 'dose_recommendation_10'] = dat.loc[i, 'coeff_0x']

        else:
            f = interpolate.interp1d(y, x, fill_value='extrapolate')

            dat.loc[i, 'dose_recommendation_8'] = f(8)
            dat.loc[i, 'dose_recommendation_10'] = f(10)

    # Create list of patients
    list_of_patients = find_list_of_patients()

    # # Create list of body weight
    # list_of_body_weight = find_list_of_body_weight()

    # # Add body weight column
    # dat['body_weight'] = ""

    # for j in range(len(dat)):
    #     index_patient = list_of_patients.index(str(dat.patient[j]))
    #     dat.loc[j, 'body_weight'] = list_of_body_weight[index_patient]
        
    # Add dose recommendations
    dat['possible_doses'] = ""
    dat['dose_recommendation'] = ""
    for i in range(len(dat)):
        # Find minimum dose recommendation by mg
        min_dose_mg = math.ceil(min(dat.dose_recommendation_8[i], dat.dose_recommendation_10[i]) * 2) / 2

        # Find maximum dose recommendation by mg
        max_dose_mg = math.floor(max(dat.dose_recommendation_8[i], dat.dose_recommendation_10[i]) * 2) / 2

        # Between and inclusive of min_dose_mg and max_dose_mg,
        # find doses that are multiples of 0.5 mg
        possible_doses = np.arange(min_dose_mg, max_dose_mg + minimum_capsule, minimum_capsule)
        possible_doses = possible_doses[possible_doses % minimum_capsule == 0]

        if possible_doses.size == 0:
            possible_doses = np.array(min(min_dose_mg, max_dose_mg))

        # Add to column of possible doses
        dat.at[i, 'possible_doses'] = possible_doses

        # Add to column of dose recommendation with lowest out of possible doses
        dat.loc[i, 'dose_recommendation'] = possible_doses if (possible_doses.size == 1) else min(possible_doses)

    CURATE_assessment = dat[['patient', 'pred_day', 'prediction', 'response', 'deviation', 'dose', 'dose_recommendation']]

    # Add columns for assessment
    CURATE_assessment['reliable'] = ""
    CURATE_assessment['accurate'] = ""
    CURATE_assessment['diff_dose'] = ""
    CURATE_assessment['therapeutic_range'] = ""
    CURATE_assessment['actionable'] = ""
    CURATE_assessment['effect_of_CURATE'] = ""

    for i in range(len(CURATE_assessment)):

        # Reliable
        if (CURATE_assessment.prediction[i] >= therapeutic_range_lower_limit) & (CURATE_assessment.prediction[i] <= therapeutic_range_upper_limit):
            prediction_range = 'therapeutic'
        else:
            prediction_range = 'non-therapeutic'

        if (CURATE_assessment.response[i] >= therapeutic_range_lower_limit) & (CURATE_assessment.response[i] <= therapeutic_range_upper_limit):
            response_range = 'therapeutic'
        else:
            response_range = 'non-therapeutic'

        reliable = (prediction_range == response_range)
        CURATE_assessment.loc[i, 'reliable'] = reliable

        # Accurate
        accurate = (dat.deviation[i] >= overprediction_limit) & (dat.deviation[i] <= underprediction_limit)
        CURATE_assessment.loc[i, 'accurate'] = accurate

        # Different dose
        diff_dose = (dat.dose[i] != dat.dose_recommendation[i])
        CURATE_assessment.loc[i, 'diff_dose'] = diff_dose

        # Therapeutic range
        therapeutic_range = (CURATE_assessment.response[i] >= therapeutic_range_lower_limit) & (CURATE_assessment.response[i] <= therapeutic_range_upper_limit)
        CURATE_assessment.loc[i, 'therapeutic_range'] = therapeutic_range

        # Actionable
        actionable = ((dat.dose_recommendation[i]) <= max_dose_recommendation) & ((dat.dose_recommendation[i]) >= min_dose_recommendation)
        CURATE_assessment.loc[i, 'actionable'] = actionable

        # Effect of CURATE
        if (reliable == True) & (accurate == True) & (diff_dose == True) & (therapeutic_range == False) & (actionable == True):
            CURATE_assessment.loc[i, 'effect_of_CURATE'] = 'Improve'
        elif (reliable == True) & (accurate == False) & (diff_dose == True) & (therapeutic_range == True) & (actionable == True):
            CURATE_assessment.loc[i, 'effect_of_CURATE'] = 'Worsen'
        elif (reliable == False) & (accurate == True) & (diff_dose == True) & (therapeutic_range == True) & (actionable == True):
            CURATE_assessment.loc[i, 'effect_of_CURATE'] = 'Worsen'
        elif (reliable == False) & (accurate == False) & (diff_dose == True) & (therapeutic_range == True) & (actionable == True):
            CURATE_assessment.loc[i, 'effect_of_CURATE'] = 'Worsen'
        else:
            CURATE_assessment.loc[i, 'effect_of_CURATE'] = 'Unaffected'

    return CURATE_assessment

# Create lists
def find_list_of_body_weight():

    xl = pd.ExcelFile(raw_data_file)
    excel_sheet_names = xl.sheet_names

    list_of_body_weight = []

    # Create list of body_weight
    for sheet in excel_sheet_names:    
        data = pd.read_excel(raw_data_file, sheet_name=sheet, index_col=None, usecols = "C", nrows=15)
        data = data.reset_index(drop=True)
        list_of_body_weight.append(data['Unnamed: 2'][13])

    list_of_body_weight = list_of_body_weight[:12]+[8.29]+list_of_body_weight[12+1:]

    return list_of_body_weight

def find_list_of_patients():
    # Declare list
    list_of_patients = []

    # Create list of patients
    wb = load_workbook(raw_data_file, read_only=True)
    list_of_patients = wb.sheetnames

    return list_of_patients

# Import data
def import_raw_data_including_non_ideal():
    df = pd.read_excel('all_data_including_non_ideal.xlsx', sheet_name='data')

    return df

# Statistical test
def result_and_distribution(df, metric_string):
    """
    Determine which normality test to do based on n.
    Conduct normality test and print results.
    Based on p-value of normality test, choose mean/median to print.
    """
    p = shapiro_test_result(df, metric_string)
        
    if p < 0.05:
        median_IQR_range(df)
    else:
        mean_and_SD(df)
        
def shapiro_test_result(df, metric_string):
    shapiro_result = stats.shapiro(df).pvalue
    
    if shapiro_result < 0.05:
        result_string = 'reject normality'
    else:
        result_string = 'assume normality'

    print(f'{metric_string}:\nShapiro test p-value = {shapiro_result:.2f}, {result_string}')
    
    return shapiro_result
    
def mean_and_SD(df):
    # Mean and SD
    mean = df.describe()['mean']
    std = df.describe()['std']
    count = df.describe()['count']

    print(f'mean {mean:.2f} | std {std:.2f} | count {count}\n')
    
def median_IQR_range(df):
    median = df.describe()['50%']
    lower_quartile = df.describe()['25%']
    upper_quartile = df.describe()['75%']
    minimum = df.describe()['min']
    maximum = df.describe()['max']
    count = df.describe()['count']

    print(f'median {median:.2f} | IQR {lower_quartile:.2f} - {upper_quartile:.2f} | count {count} | range {minimum:.2f} - {maximum:.2f}\n')

# LOOCV for all methods
def LOOCV_all_methods(file_string=result_file_total, dose='total'):
    """
    Perform LOOCV for all methods
    
    Output: Excel sheet 'all_methods_LOOCV.xlsx' with results of LOOCV for all methods
    """
    if dose == 'evening':
        file_string=result_file_evening

    dat = read_file_and_remove_unprocessed_pop_tau(file_string)

    # Define lists
    linear_patient_list = dat[dat.method.str.contains('L_')].patient.unique().tolist()
    quad_patient_list = dat[dat.method.str.contains('Q_')].patient.unique().tolist()
    method_list = dat.method.unique().tolist()

    # Keep only useful columns in dataframe
    dat = dat[['method', 'patient', 'abs_deviation']]

    # Create output dataframes
    experiment_results_df = pd.DataFrame(columns=['method', 'experiment', 'train_median', 'test_median'])
    overall_results_df = pd.DataFrame(columns=['method', 'train (median)', 'test (median)'])

    exp_res_counter = 0
    overall_res_counter = 0

    for method in method_list:

        #  Define num of patients according to whether method is linear or quadratic
        num_of_patients, patient_list = num_patients_and_list(method, linear_patient_list, quad_patient_list)

        for i in range(num_of_patients):

            train_median = find_train_median_LOOCV(dat, method, patient_list, i)
            test_median = find_test_median_LOOCV(dat, method, patient_list, i)

            # Update experiment results dataframe
            experiment_results_df.loc[exp_res_counter, 'experiment'] = i + 1
            experiment_results_df.loc[exp_res_counter, 'method'] = method
            experiment_results_df.loc[exp_res_counter, 'train_median'] = train_median
            experiment_results_df.loc[exp_res_counter, 'test_median'] = test_median

            exp_res_counter = exp_res_counter + 1

    # Find median of the train_median and test_median of each method
    train_median_df = experiment_results_df.groupby('method')['train_median'].median().reset_index()
    test_median_df = experiment_results_df.groupby('method')['test_median'].median().reset_index()

    # Create dataframe for overall results by method
    overall_results_df = train_median_df.merge(test_median_df, how='inner', on='method')

    # # Shapiro test by method, on train_median and test_median (result: some normal)
    # train_median_shapiro = experiment_results_df.groupby('method')['train_median'].apply(lambda x: stats.shapiro(x).pvalue < 0.05)
    # test_median_shapiro = experiment_results_df.groupby('method')['test_median'].apply(lambda x: stats.shapiro(x).pvalue < 0.05)

    # Output dataframes to excel as individual sheets
    with pd.ExcelWriter('LOOCV_results_' + dose + '.xlsx') as writer:
        experiment_results_df.to_excel(writer, sheet_name='Experiments', index=False)
        overall_results_df.to_excel(writer, sheet_name='Overall', index=False)

def find_test_median_LOOCV(dat, method, patient_list, i):
    """Find median of test set"""
    
    # Define test df
    test_df = dat[(dat.method == method) & (dat.patient == patient_list[i])]
    
    # Find test_median
    test_median = test_df.abs_deviation.median()
    
    return test_median

def find_train_median_LOOCV(dat, method, patient_list, i):
    """Find median of training set"""
        
    # Define train df
    train_patient_list = patient_list.copy()
    train_patient_list.pop(i)
    train_df = dat[(dat.method == method) & (dat.patient.isin(train_patient_list))]

    # Find train_median
    train_median = train_df.abs_deviation.median()
    
    return train_median

def num_patients_and_list(method, linear_patient_list, quad_patient_list):
    """Define num of patients according to whether method is linear or quadratic"""
    
    if 'L_' in method:
        num_of_patients = len(linear_patient_list)
        patient_list = linear_patient_list
    else:
        num_of_patients = len(quad_patient_list)
        patient_list = quad_patient_list
        
    return num_of_patients, patient_list

# For clinical paper
def response_vs_dose(plot=False, dose='total'):
    """
    Facetgrid scatter plot of response vs dose, colored by number of days. 

    Note: To plot color bar, uncomment commented code. 
    """
    if dose == 'total':
        all_data_file = all_data_file_total
    else:
        all_data_file = all_data_file_evening
    
    # Plot individual profiles
    dat = pd.read_excel(all_data_file)

    # Create within-range column for color
    dat['within_range'] = (dat.response <= therapeutic_range_upper_limit) & (dat.response >= therapeutic_range_lower_limit)

    # Create low/med/high dose column
    dat['dose_range'] = ""
    for i in range(len(dat)):
        if dat.dose[i] < low_dose_upper_limit:
            dat.loc[i, 'dose_range'] = 'Low'
        elif dat.dose[i] < medium_dose_upper_limit:
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
        # Plot response vs dose
        # Settings
        fig1 = plt.figure()
        sns.set(font_scale=1.5, rc={"figure.figsize": (16,10), "xtick.bottom" : True, "ytick.left" : True}, style='white')
        
        # Plot
        cmap = mpl.cm.Purples(np.linspace(0,1,20))
        cmap = mpl.colors.ListedColormap(cmap[5:,:-1])

        ax = sns.relplot(data=new_dat, x='dose', y='response', hue='Day', col='patient', col_wrap=4, style='Dose range',
                height=3, aspect=1,s=100, zorder=2, palette=cmap)

        # Add gray region for therapeutic range
        for g in ax.axes:
            g.axhspan(therapeutic_range_lower_limit, therapeutic_range_upper_limit, facecolor='grey', alpha=0.2, zorder=1)

        # Label
        ax.set_ylabels('TTL (ng/ml)')
        ax.set_titles('Patient {col_name}')
        ax.set_xlabels('Dose (mg)')
        g.set(yticks=np.arange(0,math.ceil(max(new_dat.response)),4),
        xticks=np.arange(0, max(new_dat.dose+1), step=1))

        # Legend
        ax.legend.remove()
        plt.savefig('response_vs_dose.png', dpi=1000, facecolor='w', bbox_inches='tight')

        # Colorbar
        fig2, ax2 = plt.subplots(figsize=(6, 1))
        fig2.subplots_adjust(bottom=0.4, top=0.7, hspace=.8)
        norm = mpl.colors.Normalize(vmin=0, vmax=new_dat.Day.max())

        cb = fig2.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
             cax=ax2, orientation='horizontal', label='Day')

        cb.ax.xaxis.set_label_position('top')

        legend_elements = [Line2D([0], [0], marker='o', color='w', label='Low',
                          markerfacecolor='k', markersize=9),
                          Line2D([0], [0], marker='X', color='w', label='Medium',
                          markerfacecolor='k', markersize=9),
                          Line2D([0], [0], marker='s', color='w', label='High',
                          markerfacecolor='k', markersize=9),
                          Patch(facecolor='grey', edgecolor='grey',
                          label='Region within therapeutic range', alpha=.2)]
        legend1 = plt.legend(handles=legend_elements[:3], bbox_to_anchor=(1.2,2.3), loc='upper left', frameon=False, title='Dose range')
        legend2 = plt.legend(handles=legend_elements[3:], bbox_to_anchor=(1.7,2.3), loc='upper left', frameon=False)
        ax2.add_artist(legend1)
        ax2.add_artist(legend2) 

        plt.savefig('response_vs_dose_colorbar.png', dpi=1000, facecolor='w', bbox_inches='tight')
        
    return new_dat

def extreme_prediction_errors():
    """
    Analysis of extreme prediction errors
    Output: 
    - Printed values of upper quartile of 
    absolute prediction errors, and distribution within
    extreme prediction errors from the upper quartile.
    """

    df = import_raw_data_including_non_ideal()

    # Subset RW
    df = df[df.method=='L_RW_wo_origin']

    upper_quartile = df.abs_deviation.describe().loc['75%']
    print(f'upper quartile: {upper_quartile:.2f}\n')

    # Extract predictions with deviation higher than upper quartile
    extreme_prediction_errors = df[df.abs_deviation>upper_quartile]

    print(f'distribution within extreme prediction errors: {extreme_prediction_errors["abs_deviation"].describe().loc["50%"]:.2f}\
     [IQR {extreme_prediction_errors["abs_deviation"].describe().loc["25%"]:.2f} - \
     {extreme_prediction_errors["abs_deviation"].describe().loc["75%"]:.2f}]')

    extreme_prediction_errors[['patient','pred_day','abs_deviation']]

def dosing_strategy_values():
    """
    Print the following, to file:
    - Thresholds for repeated and distributed doses
    - 1. % of patients with repeated doses
    - 2. % of days in TR when there are repeated doses
    - 3. % of patients with distributed dose
    - 4. First day where TR is achieved for distributed dose
    """
    # Uncomment stdout if not writing to file.
    original_stdout = sys.stdout
    with open('dosing_strategy_values.txt', 'w') as f:
        sys.stdout = f
        
        df = response_vs_dose()
        df = df.dropna().reset_index(drop=True)

        # Create therapeutic_range column
        for i in range(len(df)):
            if (df.response[i] >= therapeutic_range_lower_limit) & (df.response[i] <= therapeutic_range_upper_limit):
                df.loc[i, 'therapeutic_range'] = True
            else:
                df.loc[i, 'therapeutic_range'] = False

        # Find suitable lower limit of dose repeats to consider the dose strategy as repeated dosing.
        repeated_count = df.groupby('patient')['dose'].value_counts().reset_index(name='count')
        repeated_dose_threshold = repeated_count['count'].describe().loc['75%']
        print(f"Repeated dose threshold is based on {repeated_dose_threshold} which is the 75th percentile of number of repeats.\n\
            Repeated dose: > 4 dose repeats")
        

        # 1. % of patients with repeated doses
        # 2. % of days in TR when there are repeated doses

        # For each patient, identify groups of consecutive days and label a group number.
        # To do that, label the first row of each patient as group 1. 
        # For each subsequent row, if the previous row is more than 1 day apart, label it as a new group.
        df['group_num_by_repeats'] = ""
        for i in range(len(df)):
            if i==0:
                group_num = 1
            else:
                if df.patient[i] != df.patient[i-1]:
                    group_num = 1 
            
                if df.Day[i] - df.Day[i-1] > 1:
                    group_num += 1
            df.loc[i, 'group_num_by_consecutive_days'] = group_num

        # Remove unnecessary columns
        df = df[['patient', 'Day', 'dose', 'therapeutic_range','group_num_by_consecutive_days']]

        # Find number of dose repeats across consecutive days
        num_of_dose_repeats = df.groupby(['patient', 'group_num_by_consecutive_days', 'dose'])['dose'].count().reset_index(name='num_of_dose_repeats')

        # Find percentage of days in therapeutic range, for each dose in each group of consecutive days
        perc_in_therapeutic_range =  df.groupby(['patient', 'group_num_by_consecutive_days', 'dose'])['therapeutic_range'].apply(lambda x: x.sum()/x.count()*100).reset_index(name='perc_in_therapeutic_range')

        # Combine both dataframes
        combined_df = num_of_dose_repeats.merge(perc_in_therapeutic_range, on=['patient', 'group_num_by_consecutive_days', 'dose'])
        combined_df = combined_df[combined_df.num_of_dose_repeats > 4].reset_index(drop=True)

        # Print results
        perc_of_patients_with_repeated_doses = len(combined_df.patient.unique())/len(df.patient.unique())*100
        print(f"1. % of patients with repeated dose: {perc_of_patients_with_repeated_doses}% (N={len(df.patient.unique())})\n")
        result_and_distribution(combined_df.perc_in_therapeutic_range, '2. % of days in therapeutic range when there are repeated doses')
        
        # 3. % of patients with distributed dose

        # Find dose range
        dose_range = df.groupby('patient')['dose'].apply(lambda x: x.max() - x.min()).reset_index(name='dose_range')
        distributed_dose_threshold = dose_range['dose_range'].describe().loc['75%']

        print(f'Distributed dose: > {distributed_dose_threshold} mg\n')

        patients_with_distributed_dose = dose_range[dose_range.dose_range > distributed_dose_threshold].loc[:, "patient"].to_list()

        print(f'3. % of patients with distributed dose: {len(patients_with_distributed_dose)/len(df.patient.unique())*100}%,\
        {len(patients_with_distributed_dose)} out of 16 patients')
        print(f'Patients with distributed doses (> {distributed_dose_threshold} mg range): {patients_with_distributed_dose}')

        # 4. First day where TR is achieved for distributed dose
        first_day_distributed_dose = df[df.patient.isin(patients_with_distributed_dose)]
        first_day_distributed_dose = first_day_distributed_dose[first_day_distributed_dose.therapeutic_range == True]

        first_day_distributed_dose = first_day_distributed_dose.groupby('patient')['Day'].first().reset_index(name='first_day_to_TR')
        print(f'First day where TR is achieved for distributed dose: {first_day_distributed_dose.first_day_to_TR.to_list()}')

    sys.stdout = original_stdout

if __name__ == '__main__':
    # Get arguments
    import argparse
    parser = argparse.ArgumentParser(description='Plot CURATE.AI results')
    parser.add_argument("-f", "--figure", type=str, default=False)
    parser.add_argument("-a", "--analysis", type=str, default=False)
    parser.add_argument("-d", "--dose", type=str, default='total')
    args = parser.parse_args()
    
    # Figures
    if args.figure=='fig_2':
        fig_2_TTL_over_time(plot=True)
    elif args.figure=='fig_5':
        fig_5a_case_reach_TR_earlier(plot=True)
        fig_5b_case_reach_TR_earlier(plot=True)
    elif args.figure=='fig_6':
        fig_6a_case_sustain_TR_longer(plot=True)
        fig_6b_case_sustain_TR_longer(plot=True)
        fig_6c_case_sustain_TR_longer(plot=True)
    elif args.figure=='fig_7':
        effect_of_CURATE(plot=True)
        plot_SOC_CURATE_perc_in_TR()
        SOC_CURATE_first_day_in_TR(plot=True)
        SOC_CURATE_perc_pts_TR_in_first_week(plot=True)
    else:
        print('no valid figure was specified')

    # Analysis
    if args.analysis=='patient_population':
        percentage_of_pts_that_reached_TR_per_dose_range()
        patient_population_values()
    elif args.analysis=='technical_perf_metrics':
        technical_performance_metrics(dose=args.dose)
    elif args.analysis=='clinically_relevant_perf_metrics':
        clinically_relevant_performance_metrics(dose=args.dose)
    elif args.analysis=='effect_of_CURATE':
        effect_of_CURATE_categories()
        effect_of_CURATE_values()
        effect_of_CURATE_inter_indiv_differences()
    elif args.analysis=='fig_4_values':
        values_in_clinically_relevant_flow_chart()
    else:
        print('no valid analysis was specified')