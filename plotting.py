import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu
from sklearn.metrics import mean_squared_error
import math
from matplotlib import colors
from matplotlib.pyplot import cm
from matplotlib.patches import Patch
from Profile_Generation import *
from openpyxl import load_workbook
import sys
from scipy.stats import levene
from scipy.stats import wilcoxon
import statistics
from scipy.stats import bartlett

# Define file names
result_file_total = 'CURATE_results_total.xlsx'
result_file_evening = 'CURATE_results_evening.xlsx'
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
max_dose_recommendation_BW = 0.85
minimum_capsule = 0.5
therapeutic_range_upper_limit = 10
therapeutic_range_lower_limit = 8
dosing_strategy_cutoff = 0.4
acceptable_tac_upper_limit = 12
acceptable_tac_lower_limit = 6.5

def test_write_to_file(dose='total'):
    original_stdout = sys.stdout

    with open('test' + dose + '.txt', 'w') as f:
        sys.stdout = f
        a = 'test'
        b = 'boy'
        print(a)
    
    sys.stdout = original_stdout
    return b

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

def import_CURATE_results():
    df = pd.read_excel(result_file, sheet_name='result')
    return df

# Edit excel sheets
def add_body_weight_and_dose_by_body_weight_to_df_in_excel():
    
    df = pd.read_excel('all_data_including_non_ideal.xlsx')

    # Declare lists
    list_of_body_weight = find_list_of_body_weight()
    list_of_patients = find_list_of_patients()

    # Add body weight column
    df['body_weight'] = ""

    for i in range(len(df)):
        # Find index of patient in list_of_patients
        index = list_of_patients.index(str(df.patient[i]))
        body_weight = list_of_body_weight[index]    

        # Add body weight to column
        df.loc[i, 'body_weight'] = body_weight

    # Change current dose column to dose_mg
    df = df.rename(columns={'dose':'dose_mg'})

    # Add column 'dose' by dividing dose_mg by body weight
    df['dose'] = df['dose_mg'] / df['body_weight']

    with pd.ExcelWriter('all_data_including_non_ideal.xlsx', engine='openpyxl', mode='a') as writer:  
        df.to_excel(writer, sheet_name='data', index=False)

# Create excel sheets
def all_data(dose='total'):
    """
    Clean raw data and label which are ideal or non-ideal.
    Export all data to excel.
    
    Output: 
    - Dataframe of all cleaned raw data with label of ideal/non-ideal.
    - 'all_data.xlsx' with dataframe
    """
    if dose == 'total':
        dose_string = "Eff 24h Tac Dose"
    else:
        dose_string = "2nd Tac dose (pm)"

    # Create dataframe from all sheets
    list_of_patients = find_list_of_patients()
    list_of_body_weight = find_list_of_body_weight()

    df = pd.DataFrame()

    for patient in list_of_patients:
        patient_df = pd.read_excel('Retrospective Liver Transplant Data - edited.xlsx', sheet_name=patient, skiprows=17)
        patient_df['patient'] = patient

        # Subset dataframe
        patient_df = patient_df[['Day #', 'Tac level (prior to am dose)', dose_string, 'patient']]

        # Shift dose column
        patient_df[dose_string] = patient_df[dose_string].shift(1)

        # Remove "mg"/"ng" from dose
        patient_df[dose_string] = patient_df[dose_string].astype(str).str.replace('mgq', '')
        patient_df[dose_string] = patient_df[dose_string].astype(str).str.replace('mg', '')
        patient_df[dose_string] = patient_df[dose_string].astype(str).str.replace('ng', '')
        patient_df[dose_string] = patient_df[dose_string].astype(str).str.strip()
        patient_df[dose_string] = patient_df[dose_string].astype(float)

        first_day_of_dosing = patient_df['Day #'].loc[~patient_df[dose_string].isnull()].iloc[0]

        # Keep data from first day of dosing
        patient_df = patient_df[patient_df['Day #'] >= first_day_of_dosing].reset_index(drop=True)

        # Set the first day of dosing as day 2 (because first dose received on day 1)
        for i in range(len(patient_df)):
            patient_df.loc[i, 'Day #'] = i + 2

        df = pd.concat([df, patient_df])

    df.reset_index(drop=True)
    df['patient'] = df['patient'].astype(int)

    # Rename columns
    df = df.rename(columns={'Day #':'day', 'Tac level (prior to am dose)':'response', dose_string:'dose'})

    # Import output dataframe from 'clean'
    ideal_df = pd.read_excel(result_file, sheet_name='clean')

    # Add ideal column == TRUE
    ideal_df['ideal'] = True

    # Subset columns
    ideal_df = ideal_df[['day', 'patient', 'ideal']]

    # Merge dataframes
    combined_df = df.merge(ideal_df, how='left', on=['day', 'patient'])

    # Fill in ideal column with False if NaN
    combined_df['ideal'] = combined_df['ideal'].fillna(False)

    # Fill in body weight
    combined_df['body_weight'] = ""

    for i in range(len(combined_df)):
        # Find index of patient in list_of_patients
        index = list_of_patients.index(str(combined_df.patient[i]))
        body_weight = list_of_body_weight[index]    

        # Add body weight to column
        combined_df.loc[i, 'body_weight'] = body_weight

    # Add column 'dose' by dividing dose_mg by body weight
    combined_df['body_weight'] = combined_df['body_weight'].astype(float)
    combined_df['dose'] = combined_df['dose'].astype(float)
    combined_df['dose_BW'] = combined_df['dose'] / combined_df['body_weight']

    # Clean up response column
    for i in range(len(combined_df)):

        # For response column, if there is a '/', take first value
        if '/' in str(combined_df.response[i]):
            combined_df.loc[i, 'response'] = combined_df.response[i].split('/')[0]
        else: 
            combined_df.loc[i, 'response'] = combined_df.response[i]

        # If response is <2, which means contain '<', label as NaN
        if '<' in str(combined_df.response[i]):
            combined_df.loc[i, 'response'] = np.nan

    # Export to excel
    combined_df.to_excel(r'all_data_' + dose + '.xlsx', index = False, header=True)
    
    return combined_df

# Most updated code
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
    
def patient_journey_values():
    """
    Print out results of 
    1. Response
    2. % of days within therapeutic range
    3. % of participants that reached therapeutic range within first week
    4. Day where patient first achieved therapeutic range
    5. Dose administered by mg
    6. Dose administered by body weight
    """
    original_stdout = sys.stdout
    with open('patient_journey_values.txt', 'w') as f:
        sys.stdout = f
        
        data = response_vs_day(plot=False)

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
        first_week_df = first_week_df[first_week_df['Tacrolimus levels']=='Therapeutic range'].reset_index(drop=True)
        first_week_df = (first_week_df.groupby('patient')['Day'].first() <= 7).to_frame().reset_index()
        result = first_week_df.Day.sum()/first_week_df.Day.count()*100

        print(f'3. % of participants that reached therapeutic range within first week:\n{result:.2f}%,\
        {first_week_df.Day.sum()} out of 16 patients\n')

        # 4. Day where patient first achieved therapeutic range
        first_TR_df = data.copy()
        first_TR_df = first_TR_df[first_TR_df['Tacrolimus levels']=='Therapeutic range'].reset_index(drop=True)
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

def response_vs_day(file_string=all_data_file_total, plot=False, dose='total'):
    """Scatter plot of inidividual profiles, longitudinally, and response vs dose"""
    
    if dose == 'total':
        file_string = all_data_file_total
    else:
        file_string = all_data_file_evening

    print(file_string)
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
    new_dat = new_dat.rename(columns={'within_range':'Tacrolimus levels'})
    new_dat['Tacrolimus levels'] = new_dat['Tacrolimus levels'].map({True:'Therapeutic range', False: 'Non-therapeutic range'})
    new_dat = new_dat.rename(columns={'dose_range':'Dose range', 'day':'Day'})
    new_dat['patient'] = new_dat['patient'].map({84:1, 114:2, 117:3, 118:4, 120:5, 121:6, 122:7,
                                                123:8, 125:9, 126:10, 129:11, 130:12, 131:13, 132:14,
                                                133:15, 138:16})

    if plot == True:

        # Add fake row with empty data under response to structure legend columns
        new_dat.loc[len(new_dat.index)] = [2, 5, 0.5, 1, True, "", 1, "", "Low"]
        new_dat.loc[len(new_dat.index)] = [2, 5, 0.5, 1, True, "", 1, " ", "Low"]
        
        # Plot tac levels by day
        sns.set(font_scale=1.2, rc={"figure.figsize": (16,10), "xtick.bottom" : True, "ytick.left" : True}, style='white')

        g = sns.relplot(data=new_dat, x='Day', y='response', hue='Tacrolimus levels', col='patient', col_wrap=4, style='Dose range',
                height=3, aspect=1,s=80, palette=['tab:blue','tab:orange','white','white'], style_order=['Low', 'Medium', 'High', 'Unavailable'])
        
        # Add gray region for therapeutic range
        for ax in g.axes:
            ax.axhspan(therapeutic_range_lower_limit, therapeutic_range_upper_limit, facecolor='grey', alpha=0.2)
        
        g.set_titles('Patient {col_name}')
        g.set_ylabels('Tacrolimus level (ng/ml)')
        g.set(yticks=np.arange(0,math.ceil(max(new_dat.response)),4),
            xticks=np.arange(0, max(new_dat.Day+2), step=5))
        
        # Move legend below plot
        sns.move_legend(g, 'center', bbox_to_anchor=(0.18,-0.05), ncol=2)
        
        legend1 = plt.legend()
        legend_elements = [Patch(facecolor='grey', edgecolor='grey',
                            label='Region within therapeutic range', alpha=.2)]
        legend2 = plt.legend(handles=legend_elements, bbox_to_anchor=(-1.7,-0.26), loc='upper left', frameon=False)

        plt.savefig('response_vs_day_' + dose + '.png', dpi=500, facecolor='w', bbox_inches='tight')
        
        # Remove fake row before end of function
        new_dat = new_dat[:-1]

    return new_dat

def ideal_over_under_pred(file_string=result_file_total, plot=False):
    """Bar plot of percentage of ideal/over/under predictions, by method and pop tau"""
    dat = read_file_and_remove_unprocessed_pop_tau(file_string)

    # Calculate % of predictions within acceptable error, overprediction, and underprediction
    ideal = dat.groupby('method')['deviation'].apply(lambda x: ((x >= -1.5) & (x <= 2)).sum()/ x.count()*100).reset_index()
    ideal['result'] = 'ideal'
    over = dat.groupby('method')['deviation'].apply(lambda x: ((x < -1.5)).sum()/ x.count()*100).reset_index()
    over['result'] = 'over'
    under = dat.groupby('method')['deviation'].apply(lambda x: ((x >2)).sum()/ x.count()*100).reset_index()
    under['result'] = 'under'

    # Combine results into a dataframe
    metric_df = pd.concat([ideal, over, under]).reset_index(drop=True)

    # # Perform shapiro test (result: some pvalue < 0.05, some > 0.05)
    # kstest_result = metric_df.groupby(['pop_tau', 'result'])['deviation'].apply(lambda x: stats.shapiro(x).pvalue < 0.05).reset_index()

    # # Describe ideal/over/under prediction results
    # pd.set_option('display.float_format', lambda x: '%.2f' % x)
    # metric_df.groupby(['pop_tau', 'result'])['deviation'].describe()

    if plot == True:
        # Plot
        sns.set(font_scale=1.8, rc={'figure.figsize':(10,20)})
        sns.set_style("white")
        ax = sns.catplot(data=metric_df[metric_df.pop_tau == 'pop tau'], x='method', 
                        y='deviation', hue='result', kind='bar', height=5,
                        aspect=2)

        ax.set(xlabel=None, ylabel='No. of Predictions (%)', 
            title='No. of Ideal/Over/Under Predictions (%) (Pop Tau Methods)')
        ax.set_xticklabels(rotation=90)
        ax._legend.set_title('Prediction')
        plt.savefig('pop_tau_predictions.png', bbox_inches='tight', dpi=300)

        sns.set(font_scale=1.8, rc={'figure.figsize':(10,20)})
        sns.set_style("white")
        ax = sns.catplot(data=metric_df[metric_df.pop_tau == 'no pop tau'], x='method', 
                        y='deviation', hue='result', kind='bar', height=5,
                        aspect=2)
        ax.set(xlabel=None, ylabel='No. of Predictions (%)', 
            title='No. of Ideal/Over/Under Predictions (%)')
        ax.set_xticklabels(rotation=90)
        ax._legend.set_title('Prediction')
        plt.savefig('no_pop_tau_predictions.png', bbox_inches='tight', dpi=300)

    # Rename 'deviation' column to 'perc_predictions'
    metric_df.columns = ['method', 'perc_predictions', 'result']

    return metric_df

def ideal_over_under_pred_RW(plot=False):
    """Bar plot of percentage of ideal/over/under predictions, by method"""
    
    dat = ideal_over_under_pred()
    
    # Subset PPM and RW method
    dat = dat[(dat.method=='L_RW_wo_origin')]

    # Rename columns
    dat = dat.rename(columns={'result':'Result', 'method':'Method', 'perc_predictions':'Predictions (%)'})
    dat['Result'] = dat['Result'].map({'ideal':'Ideal predictions', 'over':'Over predictions', 'under':'Under predictions'})
    dat['Predictions (%)'] = dat['Predictions (%)'].round(2)

    return dat

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
        sns.set(font_scale=1.2, rc={"figure.figsize": (20,10), "xtick.bottom":True, "ytick.left":True}, style='white')
        hue_order = ['Unaffected, remain as therapeutic range', 'Unaffected, remain as non-therapeutic range',
                    'Improve to therapeutic range', 'Worsen to non-therapeutic range']
        palette = [sns.color_palette()[1], sns.color_palette()[0], sns.color_palette()[2],\
                sns.color_palette()[3]]
        style_order = ['Low', 'Medium', 'High', 'Unavailable']

        # Scatter point
        g = sns.relplot(data=combined_dat, x='day', y='response', hue='Effect of CURATE.AI-assisted dosing',\
                        hue_order=hue_order, col='patient', palette=palette,\
                        col_wrap=4, style='Dose range', height=3, aspect=1, s=80, style_order=style_order)

        # Move legend below plot
        sns.move_legend(g, 'center', bbox_to_anchor=(0.2,-0.1), title=None, ncol=2)

        # Titles and labels
        g.set_titles('Patient {col_name}')
        g.set(yticks=np.arange(0,math.ceil(max(combined_dat['response'])),4),
            xticks=np.arange(0,max(combined_dat.day),step=5))
        g.set_ylabels('Tacrolimus level (ng/ml)')
        g.set_xlabels('Day')

        # Add gray region for therapeutic range
        for ax in g.axes:
            ax.axhspan(therapeutic_range_lower_limit, therapeutic_range_upper_limit, facecolor='grey', alpha=0.2)

        legend1 = plt.legend()
        legend_elements = [Patch(facecolor='grey', edgecolor='grey',
                            label='Therapeutic range', alpha=.2)]
        legend2 = plt.legend(handles=legend_elements, bbox_to_anchor=(-1,-0.5), loc='upper left', frameon=False)

        plt.savefig('effect_of_CURATE_'+dose+'.png', dpi=1000, facecolor='w', bbox_inches='tight')

    return combined_dat

def create_df_for_CURATE_assessment(result_file = result_file_total, dose='total'):
    if dose == 'total':
        result_file = result_file_total
    else:
        result_file = result_file_evening
    
    # Import output results
    dat = pd.read_excel(result_file, sheet_name='result')
    dat_dose_by_mg = pd.read_excel(result_file, sheet_name='clean')

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
        x = np.linspace(0, max(dat.dose) + 2)
        y = p(x)
        order = y.argsort()
        y = y[order]
        x = x[order]

        dat.loc[i, 'dose_recommendation_8'] = np.interp(8, y, x)
        dat.loc[i, 'dose_recommendation_10'] = np.interp(10, y, x)

    # Create list of patients
    list_of_patients = find_list_of_patients()

    # Create list of body weight
    list_of_body_weight = find_list_of_body_weight()

    # Add body weight column
    dat['body_weight'] = ""

    for j in range(len(dat)):
        index_patient = list_of_patients.index(str(dat.patient[j]))
        dat.loc[j, 'body_weight'] = list_of_body_weight[index_patient]
        
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

    CURATE_assessment = dat[['patient', 'pred_day', 'prediction', 'response', 'deviation', 'dose', 'dose_recommendation', 'body_weight']]

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
        actionable = (dat.dose_recommendation[i]) <= max_dose_recommendation
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

        print('after open')
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

        # Add column for difference in doses recommended and administered
        df['diff_dose_recommended_and_administered'] = df['dose_recommendation'] - df['dose']

        result_and_distribution(df.diff_dose_recommended_and_administered, 'Dose recommended minus administered')
        print('done')

    sys.stdout = original_stdout

    return df

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
        sns.set(font_scale=1.2, rc={"figure.figsize": (16,10), "xtick.bottom" : True, "ytick.left" : True}, style='white')

        # Plot
        ax = sns.relplot(data=new_dat, x='dose', y='response', hue='Day', col='patient', col_wrap=4, style='Dose range',
                height=3, aspect=1,s=80)

        # Add gray region for therapeutic range
        for g in ax.axes:
            g.axhspan(therapeutic_range_lower_limit, therapeutic_range_upper_limit, facecolor='grey', alpha=0.2)

        # Label
        ax.set_ylabels('Tacrolimus level (ng/ml)')
        ax.set_titles('Patient {col_name}')
        ax.set_xlabels('Dose (mg)')

        # Legend
        ax.legend.remove()
        ax.fig.legend(handles=ax.legend.legendHandles[7:], bbox_to_anchor=(0.9,0.5), loc='center left', frameon=False)

        plt.savefig('response_vs_dose'+ dose +'.png', dpi=1000, facecolor='w', bbox_inches='tight')

        # # Colorbar
        # norm = plt.Normalize(new_dat.Day.min(), new_dat.Day.max())
        # sm = plt.cm.ScalarMappable(cmap=sns.cubehelix_palette(as_cmap=True), norm=norm)
        # sm.set_array([])
        # colorbar = ax.figure.colorbar(sm, orientation='vertical')

        # plt.savefig('response_vs_dose_colorbar.png', dpi=1000, facecolor='w', bbox_inches='tight')
        
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

def clinically_relevant_performance_metrics(result_file=result_file_total):
    """Clinically relevant performance metrics. 
    Calculate the results, conduct statistical tests, and
    print them out. 
    
    Instructions: Uncomment first block of code to write output to txt file.
    """
    # Uncomment to write output to txt file
    # file_path = 'Clinically relevant performance metrics.txt'
    # sys.stdout = open(file_path, "w")

    original_stdout = sys.stdout
    with open('clinically_relevant_perf_metrics.txt', 'w') as f:
        sys.stdout = f

        # 1. Find percentage of days within clinically acceptable 
        # tac levels (6.5 to 12 ng/ml)

        data = pd.read_excel(all_data_file)

        # Add acceptable tacrolimus levels column
        data['acceptable'] = (data['response'] >= acceptable_tac_lower_limit) & (data['response'] <= acceptable_tac_upper_limit)

        # Calculate results
        acceptable_SOC = \
        data.groupby('patient')['acceptable'].apply(lambda x: x.sum()/x.count()*100).to_frame().reset_index()

        result_and_distribution(acceptable_SOC.acceptable, '(1) % of days within clinically acceptable tac levels:')

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

        result_and_distribution(acceptable_CURATE.acceptable, '(2) % of predictions within clinically acceptable prediction error:')

        # Check normality of (1) and (2)
        SOC_shapiro_p_value = stats.shapiro(acceptable_SOC.acceptable).pvalue
        CURATE_shapiro_p_value = stats.shapiro(acceptable_CURATE.acceptable).pvalue
        print(f'SOC shapiro pvalue: {SOC_shapiro_p_value:.2f} | CURATE shapiro pvalue: {CURATE_shapiro_p_value:.2f}')
        if (SOC_shapiro_p_value > 0.05) and (CURATE_shapiro_p_value > 0.05):
            print(f'Normal, mann whitney u p-value: {mannwhitneyu(acceptable_SOC.acceptable, acceptable_CURATE.acceptable).pvalue} \n')
            print(f'Bartlett test for equal variance: {stats.bartlett(acceptable_SOC.acceptable, acceptable_CURATE.acceptable).pvalue}')
            if stats.bartlett(acceptable_SOC.acceptable, acceptable_CURATE.acceptable).pvalue > 0.05:
                print(f'Equal variance, Unpaired t-test p: {stats.ttest_ind(acceptable_SOC.acceptable, acceptable_CURATE.acceptable).pvalue:.2f}')
            else: 
                print("Unequal variance, Welch's corrected unpaired t test!! (find formula)")

        else:
            print(f'Non-normal\nmann whitney u p-value: {mannwhitneyu(acceptable_SOC.acceptable, acceptable_CURATE.acceptable).pvalue}')

        # Add unacceptable overprediction
        dat['unacceptable_overprediction'] = (dat['deviation'] < overprediction_limit)
        dat['unacceptable_underprediction'] = (dat['deviation'] > underprediction_limit)

        unacceptable_overprediction = \
        dat.groupby('patient')['unacceptable_overprediction'].apply(lambda x: x.sum()/x.count()*100).to_frame().reset_index()

        result_and_distribution(unacceptable_overprediction.unacceptable_overprediction, '(3) % clinically unacceptable overprediction')

        # 4. Clinically unacceptable underprediction

        # Add unacceptable underprediction
        unacceptable_underprediction = \
        dat.groupby('patient')['unacceptable_underprediction'].apply(lambda x: x.sum()/x.count()*100).to_frame().reset_index()

        result_and_distribution(unacceptable_underprediction.unacceptable_underprediction, '(4) % clinically unacceptable underprediction')
    
    sys.stdout = original_stdout

def technical_performance_metrics(result_file=result_file_total):
    """
    Print the following technical performance metrics
    1. Prediction erorr
    2. Absolute prediction error
    3. RMSE
    4. LOOCV
    """
    original_stdout = sys.stdout
    with open('technical_perf_metrics.txt', 'w') as f:
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
        experiment = pd.read_excel('LOOCV_results.xlsx', sheet_name='Experiments')

        experiment = experiment[experiment.method=='L_RW_wo_origin']
        result_and_distribution(experiment.train_median, 'Training set LOOCV')
        result_and_distribution(experiment.test_median, 'Test set LOOCV')
        median_IQR_range(experiment.test_median)
        print(f'Mann whitney u: {mannwhitneyu(experiment.test_median, experiment.train_median)}')

        ## Compare medians between training and test sets

    sys.stdout = original_stdout

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

        # 1. % of patients with repeated doses
        # Create therapeutic_range column
        for i in range(len(df)):
            if (df.response[i] >= therapeutic_range_lower_limit) & (df.response[i] <= therapeutic_range_upper_limit):
                df.loc[i, 'therapeutic_range'] = True
            else:
                df.loc[i, 'therapeutic_range'] = False

        # Count number of repeats
        repeated_count = df.groupby('patient')['dose'].value_counts().reset_index(name='count')
        repeated_dose_threshold = repeated_count['count'].describe().loc['75%']

        print(f'Repeated dose: > {repeated_dose_threshold} repeats\n')

        # Count number of times in therapeutic range among repeats
        repeated_therapeutic_range = df.groupby(['patient', 'dose'])['therapeutic_range'].apply(lambda x: x.sum() if x.sum() != 0 else np.nan).reset_index(name='TR')

        combined_df = repeated_count.merge(repeated_therapeutic_range, how='left', on=['patient','dose']).reset_index(drop=True)

        # Keep those with > 4 repeats
        combined_df = combined_df[combined_df['count'] > repeated_dose_threshold].reset_index(drop=True)
        combined_df = combined_df.fillna(0)

        print(f'1. % of patients with repeated dose (> {repeated_dose_threshold} repeats): \
        {(len(combined_df.patient.unique())/len(df.patient.unique())*100)}%, {len(combined_df.patient.unique())} patients')
        print(f'Patients with repeated doses (> {repeated_dose_threshold} repeats): {combined_df.patient.unique()}')

        # 2. % of days in TR when there are repeated doses
        combined_df['perc_TR_with_repeated_dose'] = combined_df['TR'] / combined_df['count'] * 100
        result_and_distribution(combined_df.perc_TR_with_repeated_dose, '\n2. % of days in TR when there are repeated doses')

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

def patient_120_day_4_recommendation(plot=False, result_file=result_file_total):
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
        x = np.linspace(0, max_dose_recommendation + 2)
        y = p(x)
        order = y.argsort()
        y = y[order]
        x = x[order]

        df.loc[i, 'dose_recommendation_8'] = np.interp(8, y, x)
        df.loc[i, 'dose_recommendation_10'] = np.interp(10, y, x)

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
        sns.set(style='white', font_scale=2,
            rc={"figure.figsize":(7,7), "xtick.bottom":True, "ytick.left":True})

        # Plot regression line
        x = np.array([combined_df.x[0],combined_df.x[1]])
        y = np.array([combined_df.y[0],combined_df.y[1]])
        a, b = np.polyfit(x, y, 1)
        x_values = np.linspace(0, 3)
        plt.plot(x_values, a*x_values + b, linestyle='-', color='y')

        # Plot scatter points
        plt.scatter(x, y, s=100, color='y')

        # Plot therapeutic range
        plt.axhspan(8, 10, facecolor='grey', alpha=0.2)

        # Label days
        for i in range(combined_df.shape[0]):
            plt.text(x=combined_df.x[i]+0.1,y=combined_df.y[i]+0.1,s=int(combined_df.day[i]),
                    fontdict=dict(color='black',size=13),
                    bbox=dict(facecolor='y', ec='black', alpha=0.5, boxstyle='circle'))

        sns.despine()
        plt.title('Day 4 Recommendation')
        plt.xlabel('Dose (mg)')
        plt.ylabel('Tacrolimus level (ng/ml)')
        plt.xticks(np.arange(0,3.5,step=0.5))
        plt.xlim(0,2.5)
        
        plt.savefig('patient_120_day_4_recommendation.png', dpi=1000, facecolor='w', bbox_inches='tight')

    return combined_df, df_original

def patient_120_response_vs_day(plot=False):
    """
    Scatter plot of response vs day for patient 120,
    with green marker for first day in therapeutic range, 
    and purple marker for potential first day with
    CURATE.AI. 
    """
    
    df, df_original = patient_120_day_4_recommendation()

    # Find predicted response on day 4
    predicted_response = (df_original.loc[0, 'coeff_1x'] * 2) + (df_original.loc[0, 'coeff_0x'])

    # SOC data
    patient_120 = pd.read_excel('all_data.xlsx')
    patient_120 = patient_120[patient_120.patient==120]
    patient_120 = patient_120[['day', 'response']].reset_index(drop=True)

    if plot==True:
        # Plot
        fig, axes = plt.subplots(figsize=(7,7))
        sns.set(style='white', font_scale=2,
            rc={"xtick.bottom":True, "ytick.left":True})

        plt.plot(patient_120.day, patient_120.response, 'yo', linestyle='-', ms=10)
        plt.scatter(x=patient_120.day[0], y=patient_120.response[0], color='y', s=100, label='Standard of care dosing')
        plt.plot(4, predicted_response, 'm^', ms=10, label='First day of therapeutic range\nwith CURATE.AI-assisted dosing')
        plt.plot(8, 9.9, 'go', ms=10, label='First day of therapeutic range\nwith standard of care dosing')

        plt.ylim(0,max(patient_120.response+1))

        sns.despine()
        plt.xticks(np.arange(2,max(patient_120.day),step=4))
        plt.xlabel('Day')
        plt.ylabel('Tacrolimus level (ng/ml)')
        plt.axhspan(8, 10, facecolor='grey', alpha=0.2)

        handles, labels = plt.gca().get_legend_handles_labels()
        order = [2,1,0]
        legend1 = plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order],
                bbox_to_anchor=(1.04,0.5), loc='center left', frameon=False) 

        legend_elements = [Patch(facecolor='grey', edgecolor='grey',
                                label='Therapeutic range', alpha=.2)]
        legend2 = plt.legend(handles=legend_elements, bbox_to_anchor=(1.04,0.34), loc='upper left', frameon=False)

        axes.add_artist(legend1)
        axes.add_artist(legend2)
        
        plt.savefig('patient_120_response_vs_day.png', dpi=1000, facecolor='w', bbox_inches='tight')

    return patient_120

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

        # 1. % of days within therapeutic range
        perc_days_within_TR = df.groupby('patient')['final_response_in_TR'].apply(lambda x: x.sum()/x.count()*100)
        perc_days_within_TR = perc_days_within_TR.reset_index(name='result')
        result_and_distribution(perc_days_within_TR.result, '1. % of days within therapeutic range')

        # 2. % of participants that reach within first week
        reach_TR_in_first_week = df[df.final_response_in_TR==True].groupby('patient')['day'].first().reset_index(name='first_day')
        reach_TR_in_first_week['result'] = reach_TR_in_first_week['first_day'] <= 7
        result = reach_TR_in_first_week['result'].sum() / len(reach_TR_in_first_week) * 100
        print(f'2. % of participants that reach within first week: {result:.2f}, {reach_TR_in_first_week["result"].sum()} out of {len(reach_TR_in_first_week)} patients\n')

        # 3. Day where patient first achieved therapeutic range
        result_and_distribution(reach_TR_in_first_week.first_day, '3. Day where patient first achieved therapeutic range')

    sys.stdout = original_stdout

    return perc_days_within_TR, reach_TR_in_first_week

def SOC_CURATE_perc_in_TR(plot=False, dose='total'):
    """
    Boxplot of % of days in TR, for SOC and CURATE.
    Print out kruskal wallis test for difference in medians.
    """

    # SOC
    perc_days_TR_SOC = response_vs_day(plot=False, dose=dose)

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

    if plot == True:
        # Plot
        sns.set(font_scale=1.2, rc={"figure.figsize": (5,5), "xtick.bottom":True, "ytick.left":True}, style='white')
        g = sns.boxplot(x="Dosing", y="Days in therapeutic range (%)", data=perc_days_TR_plot, width=0.5, palette=['#ccb974','#8172b3'])
        sns.despine()
        g.set_xlabel(None)
        g.set_ylabel('Days in therapeutic range (%)')
        g.set_xticklabels(['Standard of care\ndosing', 'CURATE.AI-assisted\ndosing'])

        # Save
        plt.savefig('SOC_CURATE_perc_in_TR_'+dose+'.png', dpi=1000, facecolor='w', bbox_inches='tight')
    
    return perc_days_TR

def barplot_SOC_CURATE_perc_in_TR():
    df = SOC_CURATE_perc_in_TR()
    sns.set(font_scale=1.2, rc={"figure.figsize": (5,5), "xtick.bottom":True, "ytick.left":True}, style='white')
    plt.bar(['Standard of care\ndosing', 'CURATE.AI-assisted\ndosing'], [mean(df.SOC), mean(df.CURATE)], yerr=[stdev(df.SOC), stdev(df.CURATE)],
        ecolor='black', capsize=10, color=['#ccb974','#8172b3'], zorder=1, width=.4)
    plt.scatter(np.zeros(len(df.SOC)), df.SOC, c='k', zorder=2)
    plt.scatter(np.ones(len(df.CURATE)), df.CURATE, c='k', zorder=3)
    for i in range(len(df.CURATE)):
        plt.plot([0,1], [df.SOC[i], df.CURATE[i]], c='k', alpha=.5)
    # plt.xticks(['Standard of care\ndosing', 'CURATE.AI-assisted\ndosing'])
    plt.ylabel('Days in therapeutic range (%)')
    sns.despine()

    plt.savefig('perc_days_in_TR.png', dpi=1000, facecolor='w', bbox_inches='tight')

def SOC_CURATE_perc_pts_TR_in_first_week(plot=False, dose='total'):
    """
    Barplot of % of patients in TR within first week, of
    SOC and CURATE.
    """
    # SOC
    data = response_vs_day(plot=False, dose=dose)

    # Drop rows where response is NaN
    data = data[data.response.notna()].reset_index(drop=True)

    # Add therapeutic range column
    for i in range(len(data)):
        if (data.response[i] >= therapeutic_range_lower_limit) & (data.response[i] <= therapeutic_range_upper_limit):
            data.loc[i, 'therapeutic_range'] = True
        else:
            data.loc[i, 'therapeutic_range'] = False

    first_week_df = data.copy()
    first_week_df = first_week_df[first_week_df['Tacrolimus levels']=='Therapeutic range'].reset_index(drop=True)
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
        sns.set(font_scale=1.2, rc={"figure.figsize": (4,5), "xtick.bottom":True, "ytick.left":True}, style='white')
        fig, ax = plt.subplots()
        ax.bar(plot_df.Dosing, plot_df.perc_reach_TR_in_first_week, width=.5, color=['#ccb974','#8172b3'])
        sns.despine()
        ax.set_xticklabels(['Standard of care\ndosing', 'CURATE.AI-assisted\ndosing'])
        plt.ylabel('Patients who achieve therapeutic\nrange in first week (%)')

        plt.savefig('SOC_CURATE_perc_pts_TR_in_first_week'+dose+'.png', dpi=1000, facecolor='w', bbox_inches='tight')

    return plot_df

def SOC_CURATE_first_day_in_TR(plot=False, dose='total'):
    """
    Boxplot for day when TR is first achieved, for
    both SOC and CURATE
    """

    # SOC
    SOC = response_vs_day(plot=False, dose=dose)
    SOC = SOC[SOC.response.notna()].reset_index(drop=True)

    # Add therapeutic range column
    for i in range(len(SOC)):
        if (SOC.response[i] >= therapeutic_range_lower_limit) & (SOC.response[i] <= therapeutic_range_upper_limit):
            SOC.loc[i, 'therapeutic_range'] = True
        else:
            SOC.loc[i, 'therapeutic_range'] = False

    SOC = SOC[SOC['Tacrolimus levels']=='Therapeutic range'].reset_index(drop=True)
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
        # Plot
        sns.set(font_scale=1.2, rc={"figure.figsize": (5,5), "xtick.bottom":True, "ytick.left":True}, style='white')
        g = sns.boxplot(x="Dosing", y="First day in therapeutic range", data=plot_df, width=0.5, palette=['#ccb974','#8172b3'])
        sns.despine()
        g.set_xlabel(None)
        # g.set_ylabel('Days in therapeutic range (%)')
        g.set_xticklabels(['Standard of care\ndosing', 'CURATE.AI-assisted\ndosing'])

        # Save
        plt.savefig('SOC_CURATE_first_day_in_TR'+dose+'.png', dpi=1000, facecolor='w', bbox_inches='tight')

    return plot_df, combined_df

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

##### New graphs after meeting with NUH ######
def cross_val():
    """ Line plot of train and test results of both K-Fold and Leave-One-Out Cross Validation for Pop Tau """
    CV_dat = pd.read_excel('GOOD OUTPUT DATA\pop_tau (by CV).xlsx', sheet_name='Overall')
    LOOCV_dat = pd.read_excel('GOOD OUTPUT DATA\pop_tau (by LOOCV).xlsx', sheet_name='Overall')

    sns.set_style("whitegrid", {'axes.grid': False} )

    ax = plt.errorbar(CV_dat.pop_tau_method, CV_dat.train_median_mean, CV_dat.train_median_SEM, linestyle='-', marker='o', color='blue', label='K-Fold CV (train)')
    ax = plt.errorbar(CV_dat.pop_tau_method, CV_dat.test_median_mean, CV_dat.test_median_SEM, linestyle='--', marker='^', color='blue', label='K-Fold CV (test)')
    ax = plt.errorbar(LOOCV_dat.pop_tau_method, LOOCV_dat.train_median_mean, LOOCV_dat.train_median_SEM, linestyle='-', marker='o', color='orange', label='LOOCV (train)')
    ax = plt.errorbar(LOOCV_dat.pop_tau_method, LOOCV_dat.test_median_mean, LOOCV_dat.test_median_SEM, linestyle='--', marker='^', color='orange', label='LOOCV (test)')
    # ax.xaxis.grid(False)

    plt.legend(bbox_to_anchor=(1.04,0.5), loc='center left')
    plt.xticks(rotation = 90)
    plt.title('Cross Validation Results')
    plt.ylabel('Absolute Prediction Error (Mean \u00b1 SEM)')
    plt.savefig('cross_val.png', dpi=300, facecolor='w', bbox_inches='tight')

def prediction_error_old(file_string='output (with pop tau by LOOCV).xlsx', plot=False):
    """ Boxplot of prediction error and absolute prediction error
    by approach, type, origin_inclusion, pop_tau."""

    dat = read_file_and_remove_unprocessed_pop_tau(file_string)

    # Add type column
    dat['type'] = ""
    for i in range(len(dat)):
        if 'L_' in dat.method[i]:
            dat.loc[i, 'type'] = 'linear'
        else:
            dat.loc[i, 'type'] = 'quadratic'

    # # Check normality of prediction error for each method (result is to reject normality)
    # for method in method_list:
    #     dat_method = dat[dat.method == method]
    #     print(stats.kstest(dat_method.deviation, 'norm').pvalue < 0.05)

    # Create approach, origin inclusion, pop tau columns
    dat['approach'] = ""
    dat['origin_inclusion'] = ""
    dat['pop_tau'] = ""
    for i in range(len(dat)):
        if 'Cum' in dat.method[i]:
            dat.loc[i, 'approach']  = 'Cum'
        elif 'PPM' in dat.method[i]:
            dat.loc[i, 'approach'] = 'PPM'
        else: dat.loc[i, 'approach'] = 'RW'
        
        if 'wo_origin' in dat.method[i]:
            dat.loc[i, 'origin_inclusion'] = 'wo_origin'
        elif 'origin_dp' in dat.method[i]:
            dat.loc[i, 'origin_inclusion'] = 'origin_dp'
        else: dat.loc[i, 'origin_inclusion'] = 'origin_int'

        if 'pop_tau' in dat.method[i]:
            dat.loc[i, 'pop_tau'] = True
        else: dat.loc[i, 'pop_tau'] = False

    if plot==True:

        # Boxplot for prediction error
        sns.set(rc={'figure.figsize':(10,7)})
        sns.set_theme(style="whitegrid", font_scale=1.4)
        ax = sns.catplot(data=dat, x='origin_inclusion', y='deviation', col='approach', hue='type', kind='box', row='pop_tau', showfliers=False)
        ax.fig.subplots_adjust(top=0.8)
        ax.fig.suptitle('Prediction Error')
        ax.set_ylabels('Prediction Error')
        plt.ylim([-15,15])
        plt.savefig('pred_error.png', bbox_inches='tight', dpi=400)

        # Boxplot for absolute prediction error
        sns.set_theme(style="whitegrid", font_scale=1.4)
        ax = sns.catplot(data=dat, x='origin_inclusion', y='abs_deviation', col='approach', hue='type', kind='box', row='pop_tau', showfliers=False)
        ax.fig.subplots_adjust(top=0.8)
        ax.fig.suptitle('Absolute Prediction Error')
        ax.set_ylabels('Prediction Error')
        plt.ylim([-5,20])
        plt.savefig('abs_pred_error.png', bbox_inches='tight', dpi=300)

    return dat

def prediction_error_PPM_RW(plot=False):
    """Boxplot of prediction error for top 2 methods."""

    dat = prediction_error()

    # Subset L_PPM_wo_origin and L_RW_wo_origin
    dat = dat[(dat.method=='L_PPM_wo_origin') | (dat.method=='L_RW_wo_origin')].reset_index()

    column_string = ['deviation', 'abs_deviation']
    ylabel_string = ['Prediction Error (ng/ml)', 'Absolute Prediction Error (ng/ml)']

    # Rename methods and column name
    dat = dat.rename(columns={column_string[0]: ylabel_string[0], 
                              column_string[1]: ylabel_string[1], 
                              'method':'Method'})
    dat['Method'] = dat['Method'].map({'L_PPM_wo_origin':'PPM', 'L_RW_wo_origin':'RW'})
    
    if plot==True:

        print('Prediction Error Plot:\n')
        
        # Set style
        sns.set(font_scale=2, rc={'figure.figsize':(6,8)})
        sns.set_style('white')
        sns.despine(top=True, right=True)

        medians = dat.groupby(['Method'])[ylabel_string[0]].median().round(2)
        vertical_offset = 0.2 # offset from median for display

        # Plot
        box_plot = sns.boxplot(data=dat, x='Method', y=ylabel_string[0], width=0.5)

        for xtick in box_plot.get_xticks():
            box_plot.text(xtick,medians[xtick] + vertical_offset,medians[xtick], 
                    horizontalalignment='center',size=15,color='w',weight='semibold')

        plt.savefig(column_string[0] + '.png', dpi=500, facecolor='w', bbox_inches='tight')

        # # Set style
        # sns.set(font_scale=2, rc={'figure.figsize':(6,8)})
        # sns.set_style('white')
        # sns.despine()

        # medians = dat.groupby(['Method'])[ylabel_string[1]].median().round(2)
        # vertical_offset = 0.2 # offset from median for display

        # # Plot
        # box_plot = sns.boxplot(data=dat, x='Method', y=ylabel_string[1], width=0.5)

        # for xtick in box_plot.get_xticks():
        #     box_plot.text(xtick,medians[xtick] + vertical_offset,medians[xtick], 
        #             horizontalalignment='center',size=15,color='w',weight='semibold')

        # plt.savefig(column_string[1] + '.png', dpi=500, facecolor='w', bbox_inches='tight')
        
    return dat

def RMSE_plot(file_string=result_file_total, plot=False):
    """
    Bar plot of RMSE for each method, grouped by pop tau and no pop tau,
    with broken y-axis
    """
    dat = read_file_and_remove_unprocessed_pop_tau(file_string)

    RMSE_method(dat)

    dat = dat.groupby('method').apply(RMSE_method).reset_index()

    # Create pop tau column and remove 'pop_tau' from method name
    dat['pop_tau'] = ""
    dat['OG_method'] = ""
    for i in range(len(dat)):
        if 'pop_tau' in dat.method[i]:
            dat.loc[i, 'pop_tau'] = 'pop tau'
            dat.loc[i, 'OG_method'] = dat.loc[i, 'method'][:-8]
        else: 
            dat.loc[i, 'pop_tau'] = 'no pop tau'
            dat.loc[i, 'OG_method'] = dat.loc[i, 'method']

    # Transform dataframe
    dat = dat[['pop_tau', 'OG_method', 'rmse']]

    # Add 'approach' column
    for i in range(len(dat)):
        if 'Cum' in dat.loc[i, 'OG_method']:
            dat.loc[i, 'approach'] = 'Cumulative'
        elif 'PPM' in dat.loc[i, 'OG_method']:
            dat.loc[i, 'approach'] = 'PPM'
        else:
            dat.loc[i, 'approach'] = 'RW'

    if plot==True:
        g = sns.catplot(data=dat, x='OG_method', y='rmse', hue='pop_tau', kind='bar', col='approach', sharex=False)
        g.set_xticklabels(rotation=90)
        g.set_ylabels('RMSE')
        plt.savefig('RMSE_by_approach.png', bbox_inches='tight', dpi=300, facecolor='w')

    return dat

def RMSE_method(dat):
    """Find RMSE by method"""
    rmse = mean_squared_error(dat.response, dat.prediction, squared=False)
    return pd.Series(dict(rmse=rmse))

def RMSE_plot_PPM_RW():
    """Barplot of RMSE for PPM and RW only"""

    dat = RMSE_plot()

    # Subset PPM and RW methods
    dat = dat[(dat.pop_tau=='no pop tau') & ((dat.OG_method=='L_PPM_wo_origin') | (dat.OG_method=='L_RW_wo_origin'))].reset_index(drop=True)

    sns.despine(top=True)
    sns.catplot(data=dat, x='approach', y= 'rmse', kind='bar', height=6, aspect=0.8)

    # Get current axis on current figure
    ax = plt.gca()

    # Iterate through the list of axes' patches
    for p in ax.patches:
        ax.text(p.get_x() + p.get_width()/2., p.get_height(), '%.2f' % float(p.get_height()), 
                fontsize=18, color='black', ha='center', va='bottom')

    plt.xlabel('Method')
    plt.ylabel('RMSE')

    plt.savefig('RMSE_PPM_RW.png', dpi=500, facecolor='w', bbox_inches='tight')
    
    return dat

def can_benefit_SOC_predictions(file_string):
    """
    Barplot of percentage of predictions that can benefit SOC, 
    by method grouped by approach, type, origin inclusion, 
    facet grouped by pop tau"""
    
    dat = read_file_and_remove_unprocessed_pop_tau(file_string)

    dat = dat[['patient', 'method', 'pred_day', 'dose', 'response', 'coeff_2x', 'coeff_1x', 'coeff_0x', 'prediction', 'deviation']]

    # # Interpolate to find percentage of possible dosing events for when prediction and observed response are outside range
    # for i in range(len(dat)):
    #     # Create function
    #     coeff = dat.loc[i, 'coeff_2x':'coeff_0x'].apply(float).to_numpy()
    #     coeff = coeff[~np.isnan(coeff)]
    #     p = np.poly1d(coeff)
    #     x = np.linspace(0, max(dat.dose)+ 2)
    #     y = p(x)
    #     order = y.argsort()
    #     y = y[order]
    #     x = x[order]

    #     dat.loc[i, 'interpolated_dose_8'] = np.interp(8, y, x)
    #     dat.loc[i, 'interpolated_dose_9'] = np.interp(9, y, x)
    #     dat.loc[i, 'interpolated_dose_10'] = np.interp(10, y, x)

    # dat[['interpolated_dose_8','interpolated_dose_9','interpolated_dose_10']].describe() # Minimum 0mg, all are possible dosing events

    # Find percentage of predictions where both observed and prediction response are outside range
    for i in range(len(dat)):
        dat.loc[i, 'both_outside'] = False
        if (round(dat.loc[i, 'prediction'],2) > 10) or (round(dat.loc[i, 'prediction'],2) < 8):
            if (round(dat.loc[i, 'response'],2) > 10) or (round(dat.loc[i, 'response'],2) < 8):
                dat.loc[i, 'both_outside'] = True

    dat['acceptable_deviation'] = (round(dat['deviation'],2) > -2) & (round(dat['deviation'],2) < 1.5)

    dat['can_benefit'] = dat['acceptable_deviation'] & dat['both_outside']

    # If can correctly identify out of range, with acceptable deviation, can benefit
    dat = dat.groupby(['method'])['can_benefit'].apply(lambda x: x.sum()/x.count() * 100).reset_index()

    # Create pop tau column and rename methods without 'pop_tau'
    dat['pop_tau'] = ""
    for i in range(len(dat)):
        if 'pop_tau' in dat.method[i]:
            dat.loc[i, 'pop_tau'] = 'pop tau'
            dat.loc[i, 'method'] = dat.method[i][:-8]
        else:
            dat.loc[i, 'pop_tau'] = 'no pop tau'
            dat.loc[i, 'method'] = dat.method[i]
            
        # Add 'approach' column
        if 'Cum' in dat.loc[i, 'method']:
            dat.loc[i, 'approach'] = 'Cumulative'
        elif 'PPM' in dat.loc[i, 'method']:
            dat.loc[i, 'approach'] = 'PPM'
        else:
            dat.loc[i, 'approach'] = 'RW'

        # Add 'type' column
        if 'L' in dat.loc[i, 'method']:
            dat.loc[i, 'type'] = 'linear'
        else:
            dat.loc[i, 'type'] = 'quadratic'

        # Add 'origin_inclusion' column
        if 'wo_origin' in dat.loc[i, 'method']:
            dat.loc[i, 'origin_inclusion'] = 'wo_origin'
        elif 'origin_dp' in dat.loc[i, 'method']:
            dat.loc[i, 'origin_inclusion'] = 'origin_dp'
        else:
            dat.loc[i, 'origin_inclusion'] = 'origin_int'

    # Shapiro test (result: no pop tau, assume normal. pop tau, reject normality.)
    # # stats.shapiro(dat[dat.pop_tau=='no pop tau'].can_benefit).pvalue       
    # # stats.shapiro(dat[dat.pop_tau=='pop tau'].can_benefit).pvalue

    # Barplot of % can benefit vs method, by method grouped by pop tau
    sns.set(font_scale=1.4)
    sns.set_style("whitegrid")

    g = sns.catplot(data=dat, x='origin_inclusion', y='can_benefit', col='approach', hue='type', row='pop_tau', kind='bar')
    g.set_axis_labels(None, "No. of Predictions that can \nPotentially Benefit SOC (%)")

    # Save
    plt.savefig('can_benefit_SOC.png', facecolor='w', dpi=300, bbox_inches='tight')

    return dat

def out_of_range():
    """Bar chart of out-of-range SOC and incorrect range prediction of CURATE."""
    origin_df = read_file_and_remove_unprocessed_pop_tau()

    # False negative/positive

    dat = origin_df[['patient', 'method', 'prediction', 'response']]

    # Create boolean, true when model predict wrong range
    for i in range(len(dat)):
        # All False
        dat.loc[i, 'wrong_range'] = False
        # Unless condition 1: prediction within range, response outside range
        if (dat.loc[i, 'prediction'] >= 8) and (dat.loc[i, 'prediction'] <= 10):
            if (dat.loc[i, 'response'] > 10) or (dat.loc[i, 'response'] < 8):
                dat.loc[i, 'wrong_range'] = True
        # Unless condition 2: prediction outside range, response within range
        elif (dat.loc[i, 'prediction'] > 10) or (dat.loc[i, 'prediction'] < 8):
            if (dat.loc[i, 'response'] >= 8) and (dat.loc[i, 'response'] <= 10):
                dat.loc[i, 'wrong_range'] = True

    dat = dat.groupby('method')['wrong_range'].apply(lambda x: x.sum() / x.count() * 100).reset_index()
    dat['source'] = 'CURATE'

    # Create another dataframe
    dat_physician = origin_df[['patient', 'method', 'prediction', 'response']]
    dat_physician = dat_physician[(dat_physician['method']=='L_Cum_wo_origin') | (dat_physician['method']=='Q_Cum_wo_origin')]
    dat_physician = dat_physician.reset_index(drop=True)

    # Create boolean, true if response is outside range
    for i in range(len(dat_physician)):
        # Set boolean default as false
        dat_physician.loc[i, 'wrong_range'] = False
        # Create boolean as True if outside range
        if (dat_physician.loc[i, 'response'] > 10) or (dat_physician.loc[i, 'response'] < 8):
            dat_physician.loc[i, 'wrong_range'] = True

    dat_physician = dat_physician.groupby('method')['wrong_range'].apply(lambda x: x.sum() / x.count() * 100).reset_index()
    dat_physician['source'] = 'SOC'

    # Create dataframe with 2 stacked dataframes of dat_physician with pop tau column for both
    # pop tau and no pop tau
    dat_physician_1 = dat_physician.copy()
    dat_physician_1['pop_tau'] = 'pop tau'
    dat_physician_2 = dat_physician.copy()
    dat_physician_2['pop_tau'] = 'no pop tau'
    dat_SOC = pd.concat([dat_physician_1, dat_physician_2]).reset_index(drop=True)

    # Rename methods to linear and quadratic only
    for i in range(len(dat_SOC)):
        if 'L_' in dat_SOC.method[i]:
            dat_SOC.loc[i, 'method'] = 'L_SOC'
        else:
            dat_SOC.loc[i, 'method'] = 'Q_SOC'

    # Create pop tau column and rename methods without 'pop_tau'
    dat['pop_tau'] = ""
    for i in range(len(dat)):
        if 'pop_tau' in dat.method[i]:
            dat.loc[i, 'pop_tau'] = 'pop tau'
            dat.loc[i, 'method'] = dat.method[i][:-8]
        else:
            dat.loc[i, 'pop_tau'] = 'no pop tau'
            dat.loc[i, 'method'] = dat.method[i]

    combined_df = pd.concat([dat, dat_SOC]).reset_index()

    # # Boxplot
    # # sns.set(font_scale=2, rc={'figure.figsize':(15,10)})
    # sns.set_theme(font_scale=2)
    # sns.set_style('whitegrid')
    # ax = sns.boxplot(data=dat, x='method', y='wrong_range', hue='source', dodge=False)
    # ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
    # ax.set_xlabel(None)
    # ax.set_ylabel('Wrong Range Predicted (%)')
    # ax.set_title('Wrong Range Predicted  (%)')
    # plt.legend(loc='upper right', bbox_to_anchor=(1.25,1))

    # Barplot
    sns.set(font_scale=1.4, rc={'figure.figsize':(5,40)})
    sns.set_style('whitegrid')

    g = sns.catplot(data=combined_df, x='method', y='wrong_range', col='pop_tau',
               kind='bar', hue='source', dodge=False)
    g.set(ylabel='No. of False Positive/\nFalse Negative Predictions (%)',
         xlabel=None)
    g.set_xticklabels(rotation=90)
    # plt.ylabels('No. of False Positive/False Negative Predictions (%)')
    # plt.xticks(rotation=90)

    plt.savefig('false_pos_neg.png', dpi=300, bbox_inches='tight')
    
    return combined_df

def OOR_predictions(file_string):
    """Barplot for OOR predictions, for both CURATE and SOC"""
    df = read_file_and_remove_unprocessed_pop_tau(file_string)

    dat = df.copy()
    # Create boolean, true when model predict wrong range
    for i in range(len(dat)):
        # All False
        dat.loc[i, 'wrong_range'] = False
        # Unless condition 1: prediction within range, response outside range
        if (dat.loc[i, 'prediction'] >= 8) and (dat.loc[i, 'prediction'] <= 10):
            if (dat.loc[i, 'response'] > 10) or (dat.loc[i, 'response'] < 8):
                dat.loc[i, 'wrong_range'] = True
        # Unless condition 2: prediction outside range, response within range
        elif (dat.loc[i, 'prediction'] > 10) or (dat.loc[i, 'prediction'] < 8):
            if (dat.loc[i, 'response'] >= 8) and (dat.loc[i, 'response'] <= 10):
                dat.loc[i, 'wrong_range'] = True

    dat = dat.groupby('method')['wrong_range'].apply(lambda x: x.sum() / x.count() * 100)
    dat = dat.to_frame().reset_index()
    dat['source'] = 'CURATE'

    # Create pop tau column and rename methods without 'pop_tau'
    dat['pop_tau'] = ""
    for i in range(len(dat)):
        if 'pop_tau' in dat.method[i]:
            dat.loc[i, 'pop_tau'] = 'pop tau'
            dat.loc[i, 'method'] = dat.method[i][:-8]
        else:
            dat.loc[i, 'pop_tau'] = 'no pop tau'
            dat.loc[i, 'method'] = dat.method[i]

    # Create another dataframe
    dat_SOC = df[['patient', 'method', 'prediction', 'response']]
    dat_SOC = dat_SOC[(dat_SOC['method']=='L_Cum_wo_origin') | (dat_SOC['method']=='Q_Cum_wo_origin')]
    dat_SOC = dat_SOC.reset_index(drop=True)

    # Create boolean, true if response is outside range
    for i in range(len(dat_SOC)):
        # Set boolean default as false
        dat_SOC.loc[i, 'wrong_range'] = False
        # Create boolean as True if outside range
        if (dat_SOC.loc[i, 'response'] > 10) or (dat_SOC.loc[i, 'response'] < 8):
            dat_SOC.loc[i, 'wrong_range'] = True

    dat_SOC = dat_SOC.groupby('method')['wrong_range'].apply(lambda x: x.sum() / x.count() * 100)
    dat_SOC = dat_SOC.to_frame().reset_index()
    dat_SOC['source'] = 'SOC'

    # Rename methods to L_SOC and Q_SOC only
    for i in range(len(dat_SOC)):
        if 'L' in dat_SOC.loc[i, 'method']:
            dat_SOC.loc[i, 'method'] = 'L_SOC'
        else:
            dat_SOC.loc[i, 'method'] = 'Q_SOC'
            
    # Create pop tau column
    dat_SOC['pop_tau'] = 'pop tau'

    # Create copy of dat_SOC with no pop tau
    dat_SOC_2 = dat_SOC.copy()
    dat_SOC_2['pop_tau'] = 'no pop tau'

    # Combine 2 dat_SOC
    dat_SOC = pd.concat([dat_SOC, dat_SOC_2])

    # Combine dat with dat_SOC
    combined_dat = pd.concat([dat, dat_SOC])

    # Boxplot for no pop tau
    sns.set(font_scale=2)
    sns.set_style('whitegrid')
    sns.catplot(data=combined_dat[combined_dat.pop_tau == 'no pop tau'], x='method', y='wrong_range', hue='source',\
                row='pop_tau', dodge=False, kind='bar', height=7, aspect=2)
    plt.ylabel('Dosing Events with \nOut-of-Range Tacrolimus Levels (%)')
    plt.xticks(rotation=90)
    plt.savefig('OOR_no_pop_tau.png', facecolor='w', dpi=300, bbox_inches='tight')

    # Boxplot for pop tau
    sns.set(font_scale=2)
    sns.set_style('whitegrid')
    sns.catplot(data=combined_dat[combined_dat.pop_tau == 'pop tau'], x='method', y='wrong_range', hue='source',\
                row='pop_tau', dodge=False, kind='bar', height=7, aspect=2)
    plt.ylabel('Dosing Events with \nOut-of-Range Tacrolimus Levels (%)')
    plt.xticks(rotation=90)
    plt.savefig('OOR_pop_tau.png', facecolor='w', dpi=300, bbox_inches='tight')

    return combined_dat

def LOOCV_all_methods_plot():
    """
    Bar plot for median absolute prediction error vs method,
    for both training and test set
    """
    
    dat = read_file_and_remove_unprocessed_pop_tau('all_methods_LOOCV.xlsx', 'Overall')
    dat = rename_methods_without_pop_tau(dat)

    dat = dat.set_index(['pop_tau', 'method'])

    # Stack dataframe
    dat = dat.stack().reset_index()

    # Rename dataframe columns
    dat.columns = ['pop_tau', 'method', 'dataset', 'median']

    # Plot for pop tau methods
    sns.set(font_scale=1.4)
    sns.set_style('whitegrid')

    g = sns.catplot(data=dat[dat.pop_tau == 'pop tau'], x='method', y='median', hue='dataset', kind='bar', sharex=False, height=5, aspect=1.5).set(title='Median Absolute Prediction Error \nfor Pop Tau Methods')
    g.set_xticklabels(rotation=90)
    g.set_ylabels('Median Absolute \nPrediction Error (ng/ml)')
    plt.ylim(0,3.5)
    plt.savefig('LOOCV_all_methods_pop_tau.png', bbox_inches='tight', dpi=300, facecolor='w')

    # Plot for no pop tau methods
    sns.set(font_scale=1.4)
    sns.set_style('whitegrid')

    g = sns.catplot(data=dat[dat.pop_tau == 'no pop tau'], x='method', y='median', hue='dataset', kind='bar', sharex=False, height=5, aspect=1.5).set(title='Median Absolute Prediction Error')
    g.set_xticklabels(rotation=90)
    g.set_ylabels('Median Absolute \nPrediction Error (ng/ml)')

    plt.ylim(0,3.5)
    plt.savefig('LOOCV_all_methods_no_pop_tau.png', bbox_inches='tight', dpi=300, facecolor='w')
    
    return dat

def LOOCV_PPM_RW(plot=False):
    """Boxplot for LOOCV results of PPM and RW only"""
    dat = pd.read_excel('all_methods_LOOCV.xlsx', sheet_name='Experiments')
    dat = dat[(dat.method=='L_PPM_wo_origin') | (dat.method=='L_RW_wo_origin')]

    dat = dat.set_index(['method', 'experiment']).stack().reset_index()
    dat.columns = ['Method','experiment','Dataset','Absolute Prediction Error']

    # Keep all columns except 'experiment'
    dat = dat[['Method', 'Dataset', 'Absolute Prediction Error']]

    dat = dat.rename(columns={'method':'Method'})
    dat['Method'] = dat['Method'].map({'L_PPM_wo_origin':'PPM', 'L_RW_wo_origin':'RW'})
    dat['Dataset'] = dat['Dataset'].map({'train_median':'Training', 'test_median':'Test'})

    if plot==True:
        # Plot
        sns.set(font_scale=1.8, style='white')
        sns.despine()

        m1 = dat.groupby(['Method']).median().round(2).values
        mL1 = [str(np.round(s, 2)) for s in m1]
        vertical_offset = 0.2 # offset from median for display

        box_plot = sns.boxplot(data=dat, x='Method', y='Absolute Prediction Error', hue='Dataset', palette='Paired')

        plt.legend(bbox_to_anchor=(1,0.5),loc='center left')

        plt.savefig('LOOCV_PPM_RW.png', dpi=500, facecolor='w', bbox_inches='tight')

    return dat

def indiv_profiles_all_data_dose_vs_response(file_string='all_data_including_non_ideal.xlsx', plot=True):
    """Scatter plot of inidividual profiles, longitudinally, and response vs dose"""
    
    # Plot individual profiles
    dat = pd.read_excel(file_string, sheet_name='clean')

    # Create within-range column for color
    dat['within_range'] = (dat.response <= 10) & (dat.response >= 8)

    # Create low/med/high dose column
    dat['dose_range'] = ""
    for i in range(len(dat)):
        if dat.dose[i] < 2:
            dat.loc[i, 'dose_range'] = 'Low Dose'
        elif dat.dose[i] < 4:
            dat.loc[i, 'dose_range'] = 'Medium Dose'
        else:
            dat.loc[i, 'dose_range'] = 'High Dose'

    # Rename columns and entries
    new_dat = dat.copy()
    new_dat = new_dat.rename(columns={'within_range':'Tacrolimus Levels'})
    new_dat['Tacrolimus Levels'] = new_dat['Tacrolimus Levels'].map({True:'Therapeutic Range', False: 'Non-therapeutic Range'})
    new_dat = new_dat.rename(columns={'dose_range':'Dose Range', 'day':'Day'})
    new_dat['patient'] = new_dat['patient'].map({84:1, 114:2, 117:3, 118:4, 120:5, 121:6, 122:7,
                                                123:8, 125:9, 126:10, 129:11, 130:12, 131:13, 132:14,
                                                133:15, 138:16})

    if plot == True:
            
        # Plot dose vs response
        sns.set(font_scale=1.2, rc={'figure.figsize':(16,10)})
        sns.set_style('white')

        # plot = plt.scatter(new_dat.dose, new_dat.response, c=new_dat.day, cmap=sns.cubehelix_palette(as_cmap=True))
        # plt.clf()
        # cbar = plt.colorbar(plot)
        # cbar.ax.tick_params(labelsize=20) 

        # plt.savefig('colorbar.png', dpi=500, facecolor='w', bbox_inches='tight')

        g = sns.relplot(data=new_dat, x='dose', y='response', hue='day', col='patient', col_wrap=4, style='Dose Range',
                height=1.5, aspect=1.5, s=60)

        g.map(plt.axhline, y=10, ls='--', c='black')
        g.map(plt.axhline, y=8, ls='--', c='black')

        # plt.colorbar(g)

        plt.savefig('indiv_pt_profile_by_dose.png', dpi=500, facecolor='w', bbox_inches='tight')
    
    return dat

def indiv_profiles_ideal_data(file_string):
    """
    Scatter plot of inidividual profiles (IDEAL DATA only), 
    longitudinally, and response vs dose
    """

    # Plot individual profiles
    dat = pd.read_excel(file_string, sheet_name='clean')

    # Create within-range column
    dat['within_range'] = (dat.response <= 10) & (dat.response >= 8)

    # Create low/med/high dose column
    dat['dose_range'] = ""
    for i in range(len(dat)):
        if dat.dose[i] < 2:
            dat.loc[i, 'dose_range'] = 'low'
        elif dat.dose[i] < 4:
            dat.loc[i, 'dose_range'] = 'medium'
        else:
            dat.loc[i, 'dose_range'] = 'high'

    sns.set(font_scale=1.2)
    sns.set_style('white')

    g = sns.relplot(data=dat[dat.ideal=='TRUE'], x='day', y='response', hue='within_range', col='patient', col_wrap=4, style='dose_range',
               height=1.5, aspect=1)

    g.map(plt.axhline, y=10, ls='--', c='black')
    g.map(plt.axhline, y=8, ls='--', c='black')

    plt.savefig('indiv_pt_profile_by_day_ideal.png', dpi=500, facecolor='w', bbox_inches='tight')

    sns.set(font_scale=1.2)
    sns.set_style('white')

    g = sns.relplot(data=dat, x='dose', y='response', hue='day', col='patient', col_wrap=4, style='dose_range',
               height=1.5, aspect=1)

    g.map(plt.axhline, y=10, ls='--', c='black')
    g.map(plt.axhline, y=8, ls='--', c='black')

    plt.savefig('indiv_pt_profile_by_dose_ideal.png', dpi=500, facecolor='w', bbox_inches='tight')
    
    return dat

def CURATE_simulated_results_both_methods():

    """
    Scatterplot for patient journey, indicating when CURATE may be useful, in terms of 
    both CURATE methods, or just one of them (PPM/RW)
    """

    dat = CURATE_could_be_useful()
    
    # Create column for adapted within range to indicate if data point
    # could have been within range if augmented by CURATE
    dat['adapted_within_range'] = dat.within_range

    dat = dat.reset_index()

    for i in range(len(dat)):
        if (dat.within_range[i]==False and dat.CURATE_could_be_useful[i]==True):
            if dat.method[i] == 'L_PPM_wo_origin':
                dat.loc[i, 'adapted_within_range'] = 'Potentially True with CURATE_PPM'
            else: 
                dat.loc[i, 'adapted_within_range'] = 'Potentially True with CURATE_RW'


    # Subset columns for combining dataframe for plotting and
    # rearrange for unstacking
    dat = dat[['pred_day', 'patient', 'method', 'adapted_within_range']]
    dat = dat.set_index(['pred_day', 'patient', 'method'])

    # Unstack
    dat = dat.unstack().reset_index()

    # Rename columns
    dat.columns = ['day', 'patient', 'PPM', 'RW']

    # Add new column for adapted_within_range
    dat['adapted_within_range'] = ""
    dat['PPM'] = dat['PPM'].astype("string")
    dat['RW'] = dat['RW'].astype("string")
    for i in range(len(dat)):

        if ('Potential' in dat.PPM[i]) and ('Potential' in dat.RW[i]):
            dat.loc[i, 'adapted_within_range'] = 'potentially_true_PPM_RW'
        elif ('Potential' in dat.PPM[i]):
            dat.loc[i, 'adapted_within_range'] = 'potentially_true_PPM'
        elif ('Potential' in dat.RW[i]):
            dat.loc[i, 'adapted_within_range'] = 'potentially_true_RW'
        else: # if no CURATE augmentation, take PPM's within range column as reference
            dat.loc[i, 'adapted_within_range'] = 'CURATE_not_helpful'

    # Only keep those that CURATE could outperform SOC
    dat = dat[dat.adapted_within_range != 'CURATE_not_helpful']
    dat = dat[['day', 'patient', 'adapted_within_range']]

    # Import data with all data including non-ideal data
    dat_all_data = indiv_profiles_all_data_day(plot=False)

    # Merge both dataframes
    combined_dat = dat_all_data.merge(dat, how='left', on=['patient', 'day'])
    combined_dat.loc[combined_dat['adapted_within_range'].isnull(),'adapted_within_range'] = \
    combined_dat['within_range']
    combined_dat['adapted_within_range'] = combined_dat['adapted_within_range'].astype(str)

    # Rename adapted_within_range
    for i in range(len(combined_dat)):
        if combined_dat.adapted_within_range[i] == 'potentially_true_PPM_RW':
            combined_dat.loc[i, 'adapted_within_range'] = 'True (PPM_RW_augmented)'
        elif combined_dat.adapted_within_range[i] == 'potentially_true_PPM':
            combined_dat.loc[i, 'adapted_within_range'] = 'True (PPM_augmented)'
        elif combined_dat.adapted_within_range[i] == 'potentially_true_RW':
            combined_dat.loc[i, 'adapted_within_range'] = 'True (RW_augmented)'

    # Plot
    sns.set(font_scale=1.2, rc={'figure.figsize':(16,10)})
    sns.set_style('white')
    hue_order = ['True', 'False', 'True (PPM_RW_augmented)', 'True (PPM_augmented)', 'True (RW_augmented)']
    palette = [sns.color_palette()[1], sns.color_palette()[0], sns.color_palette()[2],\
              sns.color_palette()[3], sns.color_palette()[4]]

    g = sns.relplot(data=combined_dat, x='day', y='response', hue='adapted_within_range',\
                    hue_order=hue_order, col='patient', palette=palette,\
                    col_wrap=4, style='dose_range', height=1.5, aspect=1.5, s=60)

    g.map(plt.axhline, y=10, ls='--', c='black')
    g.map(plt.axhline, y=8, ls='--', c='black')

    plt.savefig('indiv_pt_profile_adapted.png', dpi=500, facecolor='w', bbox_inches='tight')
    
    return combined_dat

def CURATE_simulated_results_PPM_RW():
    """
    For PPM/RW, find cases where PPM would have been useful, or harmful
    Plot simulated results. 
    """
    df = CURATE_could_be_useful()

    method_string = ['PPM', 'RW']
    method_dat = []

    for j in range(len(method_string)):

        dat = df.copy()

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
        dat_all_data = indiv_profiles_all_data_day(plot=False)

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
        
    return method_dat, method_string

def CURATE_assisted_result_distribution(method_dat, method_string):
    """    
    Plot distribution with boxplot of results in terms of percentage of dosing events where tacrolimus levels are within range
    in SOC and with CURATE, and when CURATE may be helpful or worsen the tacroliumus levels.
    
    Pre-condition: run CURATE_simulated_results_both_methods() to find method_dat and method_string
    """
    final_df_list = []

    final_df = pd.DataFrame()

    for n in range(len(method_string)):

        final_df = pd.DataFrame()

        # Find perc_dosing events for PPM/RW
        True_in_SOC = method_dat[n].groupby('patient')['adapted_within_range'].apply(lambda x: (x=='True').sum()/x.count()*100).reset_index().rename(columns={'adapted_within_range':'True_in_SOC'})

        True_after_CURATE = method_dat[n].groupby('patient')['adapted_within_range'].apply(lambda x: (x.str.count('True').sum())/x.count()*100).reset_index().rename(columns={'adapted_within_range':'True_after_CURATE'})

        CURATE_may_help = method_dat[n].groupby('patient')['adapted_within_range'].apply(lambda x: (x=='True (' + method_string[n] + '_assisted)').sum()/x.count()*100).reset_index().rename(columns={'adapted_within_range':'CURATE_may_help'})

        CURATE_may_worsen = method_dat[n].groupby('patient')['adapted_within_range'].apply(lambda x: (x=='False (' + method_string[n] + '_assisted)').sum()/x.count()*100).reset_index().rename(columns={'adapted_within_range':'CURATE_may_worsen'})

        final_df = True_in_SOC.merge(True_after_CURATE, how='left', on='patient').reset_index(drop=True)
        final_df = final_df.merge(CURATE_may_help, how='left', on='patient').reset_index(drop=True)
        final_df = final_df.merge(CURATE_may_worsen, how='left', on='patient').reset_index(drop=True)

        # Remove patient column
        final_df = final_df[['True_in_SOC', 'True_after_CURATE', 'CURATE_may_help', 'CURATE_may_worsen']]
        final_df.columns = ['True\n(SOC)', 'True\n(CURATE)', 'CURATE\nmay help', 'CURATE\nmay worsen']

        # Plot
        sns.set(font_scale=1.3)
        sns.set_style('white')
        palette = [sns.color_palette()[1], sns.color_palette()[4], sns.color_palette()[2], sns.color_palette()[3]]
        sns.catplot(data=final_df, kind='box', palette=palette, height=5, aspect=1.2)
        plt.ylabel('No. of Dosing Events (%)')
        plt.show()

        # plt.savefig(method_string[n] + '_assisted.png', dpi=500, facecolor='w', bbox_inches='tight')
        
        # print(final_df, method_dat[n])

        final_df_list = final_df_list.append(final_df)

        return final_df_list

def effect_of_CURATE_RW_old(plot=True):
    """
    Facet grid scatter plot for effect of CURATE.AI-assisted dosing on 
    therapeutic ranges
    """
    dat = clinically_relevant_flow_chart()

    method_string = ['RW']
    method_dat = []

    for j in range(len(method_string)):

        # Subset selected PPM/RW method
        dat = dat[dat['method']==('L_RW_wo_origin')]

        # Create column for adapted within range to indicate if data point
        # could have been within range if augmented by CURATE
        dat['adapted_within_range'] = dat.within_range
        dat = dat.reset_index()

        for i in range(len(dat)):
            if (dat.CURATE_could_be_useful[i]==True):
                dat.loc[i, 'adapted_within_range'] = 'Potentially True with CURATE_RW'
            elif (dat.within_range[i]==True and (dat.wrong_range[i]==True or dat.acceptable_deviation[i]==False)):
                dat.loc[i, 'adapted_within_range'] = 'Potentially False with CURATE_RW'
            else:
                dat.loc[i, 'adapted_within_range'] = 'CURATE_no_impact'

        # Subset columns
        dat = dat[['day', 'patient', 'adapted_within_range']]

        # # Rename columns
        # dat.columns = ['day', 'patient', 'adapted_within_range']

        # Only keep those that are affected by SOC
        dat = dat[dat.adapted_within_range != 'CURATE_no_impact']
        dat = dat[['day', 'patient', 'adapted_within_range']]

        # Import data with all data including non-ideal data
        dat_all_data = pd.read_excel('all_data_including_non_ideal.xlsx', sheet_name='clean')

        # Create within-range column for color
        dat_all_data['within_range'] = (dat_all_data.response <= 10) & (dat_all_data.response >= 8)

        # Create low/med/high dose column
        dat_all_data['dose_range'] = ""
        for i in range(len(dat_all_data)):
            if dat_all_data.dose[i] < 2:
                dat_all_data.loc[i, 'dose_range'] = 'Low'
            elif dat_all_data.dose[i] < 4:
                dat_all_data.loc[i, 'dose_range'] = 'Medium'
            else:
                dat_all_data.loc[i, 'dose_range'] = 'High'
                
        # Merge both dataframes
        combined_dat = dat_all_data.merge(dat, how='left', on=['patient', 'day'])
        combined_dat.loc[combined_dat['adapted_within_range'].isnull(),'adapted_within_range'] = \
        combined_dat['within_range']
        combined_dat['adapted_within_range'] = combined_dat['adapted_within_range'].astype(str)

        # Rename adapted_within_range
        for i in range(len(combined_dat)):
            if combined_dat.adapted_within_range[i] == 'Potentially True with CURATE_RW':
                combined_dat.loc[i, 'adapted_within_range'] = 'True (RW_assisted)'
            elif combined_dat.adapted_within_range[i] == 'Potentially False with CURATE_RW':
                combined_dat.loc[i, 'adapted_within_range'] = 'False (RW_assisted)'

        # Rename elements of columns
        combined_dat['dose_range'] = combined_dat['dose_range'].replace({'Low Dose':'Low', 'Medium Dose':'Medium',
                                                                        'High Dose':'High'})
        combined_dat['adapted_within_range'] = combined_dat['adapted_within_range'].replace({'False':'Unaffected, remain as non-therapeutic range',
                                                                                            'True':'Unaffected, remain as therapeutic range',
                                                                                            'True (RW_assisted)':'Improve to therapeutic range',
                                                                                            'False (RW_assisted)':'Worsen to non-therapeutic range'})
        combined_dat = combined_dat.rename(columns={'adapted_within_range':'Effect of CURATE.AI-assisted dosing', 'dose_range':'Dose range',
                                                'day':'Day', 'response':'Tacrolimus level (ng/ml)'})
        combined_dat['patient'] = combined_dat['patient'].map({84:1, 114:2, 117:3, 118:4, 120:5, 121:6, 122:7,
                                                    123:8, 125:9, 126:10, 129:11, 130:12, 131:13, 132:14,
                                                    133:15, 138:16})

        if plot==True:
            # Plot
            sns.set(font_scale=1.2, rc={"figure.figsize": (20,10), "xtick.bottom":True, "ytick.left":True}, style='white')
            hue_order = ['Unaffected, remain as therapeutic range', 'Unaffected, remain as non-therapeutic range',
                        'Improve to therapeutic range', 'Worsen to non-therapeutic range']
            palette = [sns.color_palette()[1], sns.color_palette()[0], sns.color_palette()[2],\
                    sns.color_palette()[3]]

            # Scatter point
            g = sns.relplot(data=combined_dat, x='Day', y='Tacrolimus level (ng/ml)', hue='Effect of CURATE.AI-assisted dosing',\
                            hue_order=hue_order, col='patient', palette=palette,\
                            col_wrap=4, style='Dose range', height=3, aspect=1, s=80)

            # Move legend below plot
            sns.move_legend(g, 'center', bbox_to_anchor=(0.2,-0.1), title=None, ncol=2)

            # Titles and labels
            g.set_titles('Patient {col_name}')
            g.set(yticks=np.arange(0,math.ceil(max(combined_dat['Tacrolimus level (ng/ml)'])),4),
                xticks=np.arange(0,max(combined_dat.Day),step=5))
            g.set_ylabels('Tacrolimus level (ng/ml)')

            # Add gray region for therapeutic range
            for ax in g.axes:
                ax.axhspan(8, 10, facecolor='grey', alpha=0.2)

            legend1 = plt.legend()
            legend_elements = [Patch(facecolor='grey', edgecolor='grey',
                                label='Therapeutic range', alpha=.2)]
            legend2 = plt.legend(handles=legend_elements, bbox_to_anchor=(-1,-0.5), loc='upper left', frameon=False)

            plt.savefig('indiv_pt_profile_adapted_RW.png', dpi=500, facecolor='w', bbox_inches='tight')

    return combined_dat

def read_file_and_remove_unprocessed_pop_tau(file_string=result_file_total, sheet_string='result'):
    dat = pd.read_excel(file_string, sheet_name=sheet_string)

    # Keep all methods in dataframe except strictly tau methods (contains 'tau' but does not contain 'pop')
    method_list = dat.method.unique().tolist()
    exclude_method_list = [x for x in method_list if (('tau' in x) and ('pop' not in x))]
    method_list = [x for x in method_list if x not in exclude_method_list]
    dat = dat[dat.method.isin(method_list)]
    dat = dat.reset_index(drop=True)

    return dat

def rename_methods_without_pop_tau(dat):
    """Create pop tau column and rename methods without 'pop_tau'"""
    dat['pop_tau'] = ""
    for i in range(len(dat)):
        if 'pop_tau' in dat.method[i]:
            dat.loc[i, 'pop_tau'] = 'pop tau'
            dat.loc[i, 'method'] = dat.method[i][:-8]
        else:
            dat.loc[i, 'pop_tau'] = 'no pop tau'
            dat.loc[i, 'method'] = dat.method[i]

    return dat

def case_series_118(plot, dose, result_file):
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

def case_series_118_repeated_dosing_multiple_plots(plot=False, dose='total'):
    """
    Multiple plots of response vs dose for repeated dosing strategy of 
    patient 118, with each plot representing one day of prediction. 
    """
    if dose == 'total':
        result_file = result_file_total
    else:
        result_file = result_file_evening

    dat_original, combined_df = case_series_118(plot, dose, result_file)

    # Subset repeated doses
    combined_df = combined_df[(combined_df.pred_day > 5) & (combined_df.pred_day < 10)].reset_index(drop=True)

    sns.set(style='white', font_scale=2,
        rc={"xtick.bottom":True, "ytick.left":True})

    fig, ax = plt.subplots(1, 5, figsize=(25,7))

    # Loop through number of predictions chosen
    for i in range(4):

        plt.subplot(1,5,i+1)

        # Plot regression line
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
            plt.ylabel('Tacrolimus level (ng/ml)')
        plt.yticks(np.arange(0,15,step=2))
        plt.xlabel('Dose (mg)')
        plt.xticks(np.arange(0,9,step=2))
        plt.title('Day ' + str(combined_df.pred_day[i*2+1]) + ' recommendation', size=22)

        # Label days
        for j in range(2):
            plt.text(x=combined_df.x[i*2+j]+0.4,y=combined_df.y[i*2+j]+0.4,s=int(combined_df.day[i*2+j]), 
                fontdict=dict(color='black',size=16),
                bbox=dict(facecolor='y', ec='black', alpha=0.5, boxstyle='circle'))
            
        # Add legend for grey patch of therapeutic range
        if i == 0:
            legend_elements = [Patch(facecolor='grey', edgecolor='grey',
                                    label='Therapeutic range', alpha=.2)]
            plt.legend(handles=legend_elements, bbox_to_anchor=(-0.2,-.3), loc='upper left', frameon=False)       

    plt.tight_layout()
    plt.savefig('patient_118_case_series_repeated_' + dose + '.png',dpi=500)

    return combined_df

def case_series_118_repeated_dosing_single_plot():
    """
    Single plot plots of response vs dose for repeated dosing strategy of patient 118.
    """
    dat_original, combined_df = case_series_118()
    
    # Subset repeated doses
    combined_df = combined_df[(combined_df.pred_day > 4) & (combined_df.pred_day < 11)].reset_index(drop=True)
    
    color = iter(cm.YlOrBr(np.linspace(0.3, 1, 6)))

    sns.set(style='white', font_scale=1.5,
           rc={"figure.figsize":(8,5), "xtick.bottom":True, "ytick.left":True})

    # Plot grey therapeutic range
    plt.axhspan(8, 10, facecolor='lightgrey', alpha=0.8, zorder=1)

    for i in range(6):

        # Fit dose and responses on each day of prediction to linear regression model
        x = np.array([combined_df.x[i*2],combined_df.x[i*2+1]])
        y = np.array([combined_df.y[i*2],combined_df.y[i*2+1]])
        a, b = np.polyfit(x, y, 1)

        # Plot linear regression line
        x_values = np.linspace(0, 9)
        c = next(color)
        plt.plot(x_values, a*x_values + b, linestyle='-', color=c, label=('Day '+str(combined_df.pred_day[i*2+1])))

        # Plot data points of each day
        plt.scatter(x, y, s=100, color=c, zorder=2)

        sns.despine()
        plt.ylim(0,max(combined_df.y+2))
        plt.ylabel('Tacrolimus level (ng/ml)')
        plt.yticks(np.arange(0,15,step=2))
        plt.xlabel('Dose (mg)')
        plt.xticks(np.arange(0,9,step=2))

    # Label each data point with corresponding day
    for i in range(combined_df.shape[0]):
        plt.text(x=combined_df.x[i]+0.7,y=combined_df.y[i]+0.7,s=int(combined_df.day[i]), 
        fontdict=dict(color='black',size=14),
        bbox=dict(facecolor='white', ec='black', alpha=0.5, boxstyle='circle'))

    legend = plt.legend(title='Dose recommendation for:', bbox_to_anchor=(1.04,.5), loc='center left', 
                        frameon=False, fontsize=14)
    legend.get_title().set_fontsize('16') 

    # Save
    plt.tight_layout()
    plt.savefig('patient_118_case_series_repeated_one.png',dpi=500)
    
    return combined_df

def case_series_118_repeated_dosing_dose_vs_day(plot=False, dose='total'):
    """Line and scatter plot of repeated dose vs day for CURATE.AI-assisted vs SOC"""
    if dose == 'total':
        result_file = result_file_total
    else:
        result_file = result_file_evening

    dat_original, combined_df = case_series_118(plot, dose, result_file)
    clean_dat = pd.read_excel(result_file, sheet_name='clean')
    
    # Subset pred_days with repeated dose of 6mg
    dat = dat_original[(dat_original.pred_day >= 6) & (dat_original.pred_day <= 9)]
    CURATE_dosing = [5.5, 5.5, 5, 5]
    dat['CURATE.AI-assisted dosing'] = CURATE_dosing

    # Subset columns
    dat = dat[['pred_day','CURATE.AI-assisted dosing']]
    dat = dat.rename(columns={'pred_day':'day'})

    # Subset patient 118 data only
    clean_dat = clean_dat[(clean_dat.patient == 118) & ((clean_dat.day >= 5) & (clean_dat.day <= 9))].reset_index(drop=True)

    # Subset day and dose
    clean_dat = clean_dat[['day', 'dose']]
    clean_dat = clean_dat.rename(columns={'dose':'Standard of care dosing'})

    # Combine both CURATE.AI-assisted dosing recommendations and actual dose given
    combined_dat = dat.merge(clean_dat, how='left', on='day')

    # Plot
    sns.set(font_scale=2, rc={"figure.figsize": (7,7), "xtick.bottom":True, "ytick.left":True}, style='white')

    plt.plot(combined_dat['day'], combined_dat['CURATE.AI-assisted dosing'], marker='^', color='m', label='CURATE.AI-assisted dosing', ms=10)
    plt.plot(combined_dat['day'], combined_dat['Standard of care dosing'], marker='o', color='y', label='Standard of care dosing', ms=10)
    plt.legend(bbox_to_anchor=(0.5,-0.5), loc='center', frameon=False)
    sns.despine()
    plt.xlabel('Day')
    plt.ylabel('Dose (mg)')
    plt.yticks(np.arange(5,8,step=0.5))
    plt.ylim(0, 6.5)
    

    plt.tight_layout()
    plt.savefig('patient_118_repeated_dose_dose_vs_day_'+ dose +'.png',dpi=1000)

    return combined_dat, dat_original

def case_series_118_repeated_dosing_response_vs_dose():
    """Scatter plot of dose and response for CURATE.AI-assisted and SOC dosing"""
    dat_original, combined_df = case_series_118()
    clean_dat = pd.read_excel(result_file, sheet_name='clean')

    # Subset pred_days with repeated dose of 6mg
    dat = dat_original[(dat_original.pred_day >= 6) & (dat_original.pred_day <= 9)].reset_index(drop=True)

    # Add column for CURATE recommendation
    CURATE_dosing = [5.5, 5.5, 5, 5]
    dat['CURATE-recommended dose'] = CURATE_dosing

    # Add column for predicted response if CURATE dose was administered instead
    dat['predicted_response_based_on_rec'] = dat['coeff_1x'] * dat['CURATE-recommended dose'] + dat['coeff_0x']

    dat = dat[['pred_day', 'dose', 'response', 'CURATE-recommended dose', 'predicted_response_based_on_rec']]

    # Plot
    fig, axes = plt.subplots()
    sns.set(font_scale=2, rc={"figure.figsize": (7,7), "xtick.bottom":True, "ytick.left":True}, style='white')

    plt.scatter(x=dat['CURATE-recommended dose'], y=dat['predicted_response_based_on_rec'], marker='^', color='m', label='CURATE.AI-assisted dosing', s=100)
    plt.scatter(x=dat['dose'], y=dat['response'], marker='o', color='y', label='Standard of care dosing', s=100)
    sns.despine()
    plt.xlabel('Dose (mg)')
    plt.ylabel('Tacrolimus level (ng/ml)')
    plt.axhspan(8, 10, facecolor='grey', alpha=0.2)
    plt.xticks(np.arange(4,8.5,step=0.5))
    plt.xlim(4,8)
    plt.yticks(np.arange(8,15,step=1))

    legend1 = plt.legend(bbox_to_anchor=(0.5,-0.3), loc='center', frameon=False)

    legend_elements = [Patch(facecolor='grey', edgecolor='grey',
                            label='Therapeutic range', alpha=.2)]
    legend2 = plt.legend(handles=legend_elements, bbox_to_anchor=(0,-0.33), loc='upper left', frameon=False)

    
    axes.add_artist(legend1)
    axes.add_artist(legend2)        

    for i in range(dat.shape[0]):
        plt.text(x=dat.dose[i]+0.1,y=dat.response[i]+0.1,s=int(dat.pred_day[i]),
                 fontdict=dict(color='black',size=13),
                 bbox=dict(facecolor='y', ec='black', alpha=0.5, boxstyle='circle'))

        plt.text(x=dat.loc[i, 'CURATE-recommended dose']+0.2,y=dat.loc[i, 'predicted_response_based_on_rec']+0.2,s=int(dat.pred_day[i]),
             fontdict=dict(color='black',size=13),
             bbox=dict(facecolor='m', ec='black', alpha=0.5, boxstyle='circle'))

    plt.tight_layout()
    plt.savefig('patient_118_repeated_dose_response_vs_dose.png',dpi=1000, bbox_inches='tight')
    
    return dat

def boxplot_first_day_to_therapeutic_range():
    """
    Create boxplot for first day to achieve therapeutic range for both
    SOC and CURATE.AI-assisted dosing
    """
    dat = effect_of_CURATE_RW(plot=False)


    # Compute values
    SOC = dat[dat.within_range==True].groupby('patient').first().Day.reset_index()

    CURATE = dat[(dat['Effect of CURATE.AI-assisted dosing']=='Improve to therapeutic range') | (dat['Effect of CURATE.AI-assisted dosing']=='Unaffected, remain as therapeutic range')]
    CURATE = CURATE.groupby('patient').first().Day.reset_index()

    # Add 'dosing' column to both dataframes
    SOC['dosing'] = 'Standard of care\ndosing'
    CURATE['dosing'] = 'CURATE.AI-assisted\ndosing'

    # Concat dataframes
    combined_dat = pd.concat([SOC, CURATE])

    # Boxplot
    sns.set(font_scale=1.2, rc={"figure.figsize": (5,5), "xtick.bottom":True, "ytick.left":True}, style='white')
    g = sns.boxplot(x="dosing", y="Day", data=combined_dat, width=0.5, palette=['#ccb974','#8172b3'])
    sns.despine()
    g.set_xlabel(None)
    g.set_ylabel('First day to reach therapeutic range')

    # stats.kruskal(SOC.Day, CURATE.Day).pvalue

    plt.savefig('effect_of_CURATE_first_days_median.png', dpi=500, facecolor='w', bbox_inches='tight')
    
    return combined_dat

def barplot_first_day_to_achieve_therapeutic_range_per_patient():
    
    new_dat = boxplot_first_day_to_therapeutic_range()

    new_dat['dosing'] = new_dat['dosing'].replace({'Standard of care\ndosing':'Standard of care dosing', 
                                                 'CURATE.AI-assisted\ndosing':'CURATE.AI-assisted dosing'})

    sns.set(font_scale=1.2, rc={"figure.figsize": (20,5), "xtick.bottom":True, "ytick.left":True}, style='white')
    sns.catplot(data=new_dat, x='patient', y='Day', hue='dosing', kind='bar',
               palette=['#ccb974','#8172b3'], legend=None)

    plt.legend(bbox_to_anchor=(1.04,0.5), frameon=False)
    plt.xlabel('Patient')
    plt.ylabel('First day to reach therapeutic range')

    plt.savefig('effect_of_CURATE_first_day_per_patient.png', dpi=500, facecolor='w', bbox_inches='tight')
    
    return new_dat

def barplot_percentage_days_in_therapeutic_range():
    dat = effect_of_CURATE_RW(plot=False)

    # Bar graph with percentage of days within therapeutic range for SOC and CURATE
    SOC = (dat['within_range'].value_counts()[True]) / dat['within_range'].count() * 100

    # Compute values
    CURATE_therapeutic_range = dat['within_range'].sum() + \
    dat['Effect of CURATE.AI-assisted dosing'].value_counts()['Improve to therapeutic range'] - \
    dat['Effect of CURATE.AI-assisted dosing'].value_counts()['Worsen to non-therapeutic range']
    CURATE = CURATE_therapeutic_range / dat['Effect of CURATE.AI-assisted dosing'].count() * 100

    # Create dataframe with vales
    plot_data = pd.Series({'Standard of care\ndosing': SOC, 'CURATE.AI-assisted\ndosing':CURATE}).to_frame().reset_index()
    plot_data.columns = ['Dosing', 'Days within therapeutic range (%)']
    plot_data['Days within therapeutic range (%)'] = plot_data['Days within therapeutic range (%)'].round(2)

    # Plot
    sns.set(font_scale=1.2, rc={"figure.figsize": (4,5), "xtick.bottom":True, "ytick.left":True}, style='white')
    fig, ax = plt.subplots()
    bars = ax.bar(plot_data['Dosing'], plot_data['Days within therapeutic range (%)'], width=0.5, color=['y', 'm'])

    # Label bars
    for bars in ax.containers:
        ax.bar_label(bars, fontsize=13)

    # Aesthetics
    plt.ylabel('Days within therapeutic range (%)')
    sns.despine()

    plt.savefig('effect_of_CURATE_TTR_days_all.png', dpi=500, facecolor='w', bbox_inches='tight')
    
    return plot_data

def boxplot_percentage_days_in_therapeutic_range():
    
    dat = effect_of_CURATE_RW(plot=False)

    # Compute values
    SOC = dat.groupby('patient')['within_range'].apply(lambda x: x.sum()/x.count()*100).to_frame().reset_index()
    SOC['dosing'] = 'Standard of care\ndosing'

    within_range = dat.groupby('patient')['within_range'].sum()
    CURATE_effect = dat.groupby('patient')['Effect of CURATE.AI-assisted dosing'].apply(lambda x: (x=='Improve to therapeutic range').sum() - (x=='Worsen to non-therapeutic range').sum())
    total = dat.groupby('patient')['within_range'].count()
    CURATE = ((within_range + CURATE_effect) / total * 100).to_frame().reset_index()
    CURATE['dosing'] = 'CURATE.AI-assisted\ndosing'
    CURATE = CURATE.rename(columns={0:'Effect of CURATE.AI-assisted dosing'})

    # Rename columns
    SOC = SOC.rename(columns={'within_range':'Days within therapeutic range (%)'})
    CURATE = CURATE.rename(columns={'Effect of CURATE.AI-assisted dosing':'Days within therapeutic range (%)'})

    combined_data = pd.concat([SOC, CURATE])

    # Boxplot
    sns.set(font_scale=1.2, rc={"figure.figsize": (5,5), "xtick.bottom":True, "ytick.left":True}, style='white')
    g = sns.boxplot(x="dosing", y="Days within therapeutic range (%)", data=combined_data, width=0.5, palette=['#ccb974','#8172b3'])
    sns.despine()
    g.set_xlabel(None)

    plt.savefig('effect_of_CURATE_TTR_days_median.png', dpi=500, facecolor='w', bbox_inches='tight')
    
    return combined_data

def barplot_percentage_days_therapeutic_range_per_patient():
    
    new_dat = barplot_percentage_days_in_therapeutic_range()

    new_dat['dosing'] = new_dat['dosing'].replace({'Standard of care\ndosing':'Standard of care dosing', 
                                                 'CURATE.AI-assisted\ndosing':'CURATE.AI-assisted dosing'})

    sns.set(font_scale=1.2, rc={"figure.figsize": (20,5), "xtick.bottom":True, "ytick.left":True}, style='white')
    sns.catplot(data=new_dat, x='patient', y='Days within therapeutic range (%)', hue='dosing', kind='bar',
               palette=['#ccb974','#8172b3'], legend=None)

    plt.legend(bbox_to_anchor=(1.04,0.5), frameon=False)
    plt.xlabel('Patient')

    plt.savefig('effect_of_CURATE_per_patient.png', dpi=500, facecolor='w', bbox_inches='tight')
    
    return new_dat

# LOOCV for all methods

def LOOCV_all_methods(file_string=result_file_total):
    """
    Perform LOOCV for all methods
    
    Output: Excel sheet 'all_methods_LOOCV.xlsx' with results of LOOCV for all methods
    """
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
    with pd.ExcelWriter('LOOCV_results.xlsx') as writer:
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

##### ANALYSIS

def CURATE_could_be_useful(file_string=result_file_total):
    """
    Exclude cases where CURATE cannot be u  seful for top 2 methods (PPM and RW), and
    keep those that are useful.
    
    Output: Dataframe describing results
    """
    dat = read_file_and_remove_unprocessed_pop_tau()
    dat = dat[['patient', 'method', 'pred_day', 'dose', 'response', 'coeff_2x', 'coeff_1x', 'coeff_0x', 'prediction', 'deviation']]

    # Subset CURATE models
    dat = dat[(dat.method=='L_PPM_wo_origin') | (dat.method=='L_RW_wo_origin')]

    dat = dat.reset_index(drop=True)

    # Interpolate
    for i in range(len(dat)):
        # Create function
        coeff = dat.loc[i, 'coeff_2x':'coeff_0x'].apply(float).to_numpy()
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

    interpolation = dat[['interpolated_dose_8','interpolated_dose_9','interpolated_dose_10']].describe() # Minimum 0mg, all are possible dosing events

    # Create boolean, true when model predict wrong range
    for i in range(len(dat)):
        # All False
        dat.loc[i, 'wrong_range'] = False
        # Unless condition 1: prediction within range, response outside range
        if (dat.loc[i, 'prediction'] >= 8) and (dat.loc[i, 'prediction'] <= 10):
            if (dat.loc[i, 'response'] > 10) or (dat.loc[i, 'response'] < 8):
                dat.loc[i, 'wrong_range'] = True
        # Unless condition 2: prediction outside range, response within range
        elif (dat.loc[i, 'prediction'] > 10) or (dat.loc[i, 'prediction'] < 8):
            if (dat.loc[i, 'response'] >= 8) and (dat.loc[i, 'response'] <= 10):
                dat.loc[i, 'wrong_range'] = True

    dat['acceptable_deviation'] = (round(dat['deviation'],2) >= -1.5) & (round(dat['deviation'],2) <= 2)

    # dat = dat.reset_index(drop=True)

    # Find number of predictions in wrong range by group
    wrong_range = dat.groupby('method')['wrong_range'].sum()

    # Find number of predictions with unacceptable deviations by group
    unacceptable_dev = dat.groupby('method')['acceptable_deviation'].apply(lambda x: x.count()-x.sum())

    # Find difference between interpolated dose for 9ng/ml and dose prescribed
    dat['diff_dose'] = dat['interpolated_dose_9'] - dat['dose']
    dat['abs_diff_dose'] = abs(dat['diff_dose'])

    # Create reasonable dose column
    dat['reasonable_dose'] = dat['abs_diff_dose'] >= 0.5

    unreasonable_dose = dat.groupby('method')['reasonable_dose'].apply(lambda x: x.count()-x.sum())

    # Create column for within range
    dat['within_range'] = (dat['response'] <= 10) & (dat['response'] >= 8)

    within_range = dat.groupby('method')['within_range'].sum()

    dat['CURATE_could_be_useful'] = (dat.acceptable_deviation==True) & \
        (dat.wrong_range==False) & \
            (dat.reasonable_dose==True) & \
                (dat.within_range==False)

    # # Keep only predictions with acceptable deviations
    # dat = dat[dat.acceptable_deviation==True]

    # # Keep only predictions with right range
    # dat = dat[dat.wrong_range==False]

    # # Keep reasonable doses only
    # dat = dat[dat.reasonable_dose==True]

    # # Keep those outside range
    # dat = dat[dat.within_range==False]

    # dat.groupby('method')['diff_dose'].describe().T.applymap('{:,.2f}'.format)

    return dat

def clinically_relevant_flow_chart_old(result_file=result_file_total):

    dat = CURATE_could_be_useful()

    # Subset RW
    dat = dat[dat.method=='L_RW_wo_origin'].reset_index(drop=True)

    # Find number of wrong range predictions
    number_of_unreliable_predictions = dat['wrong_range'].sum()

    # # Keep reliable predictions
    # dat = dat[dat.wrong_range==False].reset_index(drop=True)

    # Find number of inaccurate predictions with clinically acceptable prediction error
    number_of_inaccurate_predictions = len(dat) - dat.acceptable_deviation.sum()

    # # Keep accurate predictions
    # dat = dat[dat.acceptable_deviation==True].reset_index(drop=True)

    # Check if recommended doses are less than 0.55mg/kg/day
    dat['reasonable_dose'] = True
    for i in range(len(dat)):
        dat.loc[i, 'reasonable_dose'] = min(dat.interpolated_dose_8[i], dat.interpolated_dose_9[i], dat.interpolated_dose_10[i]) < 0.55

    number_of_unreasonable_doses = len(dat) - dat.reasonable_dose.sum()

    # # Keep reasonable doses
    # dat = dat[dat.reasonable_dose==True].reset_index(drop=True)

    ## Change pred_day to day 
    dat = dat.rename(columns={'pred_day':'day'})

    # Add original dose column
    clean_data = pd.read_excel(result_file, sheet_name='clean')
    combined_data = dat.merge(clean_data[['day', 'patient', 'dose_mg']], how='left', on=['patient', 'day'])

    # Declare lists
    list_of_patients = []
    list_of_body_weight = []

    # Create list of patients
    wb = load_workbook('Retrospective Liver Transplant Data.xlsx', read_only=True)
    list_of_patients = wb.sheetnames
    wb.close()

    # Create list of body_weight
    for i in range(len(list_of_patients)):    
        data = pd.read_excel('Retrospective Liver Transplant Data.xlsx', list_of_patients[i], index_col=None, usecols = "C", nrows=15)
        data = data.reset_index(drop=True)
        list_of_body_weight.append(data['Unnamed: 2'][13])

    list_of_body_weight = list_of_body_weight[:12]+[8.29]+list_of_body_weight[12+1:]

    # Add body weight column
    combined_data['body_weight'] = ""
    for j in range(len(combined_data)):
        index_patient = list_of_patients.index(str(combined_data.patient[j]))
        combined_data.loc[j, 'body_weight'] = list_of_body_weight[index_patient]
        
    combined_data['interpolated_dose_8_mg'] = combined_data['interpolated_dose_8'] * combined_data['body_weight']
    combined_data['interpolated_dose_9_mg'] = combined_data['interpolated_dose_9'] * combined_data['body_weight']
    combined_data['interpolated_dose_10_mg'] = combined_data['interpolated_dose_10'] * combined_data['body_weight']

    combined_data[['interpolated_dose_8_mg', 'interpolated_dose_9_mg', 'interpolated_dose_10_mg']]

    # recommended_dose_mg = [2.5, 2.5, 4.5, 5.5, 5, 5, 2, 2.5, 4.5, 4.5, np.nan, np.nan, 0, 6, 1.5, 2, 2.5, 3.5, 3.5, 2, 0,
    #                     1.5, 1.5, np.nan, np.nan, np.nan, np.nan, 2.5, 2.5, 0, np.nan, 3, np.nan, 0.5, 0, 0, 2.5, 2.5, 3]

    # combined_data['recommended_dose_mg'] = recommended_dose_mg

    # combined_data['diff_dose_mg'] = combined_data['dose_mg'] - combined_data['recommended_dose_mg']
    # combined_data['abs_diff_dose_mg'] = abs(combined_data['dose_mg'] - combined_data['recommended_dose_mg'])
    # combined_data['diff_dose_mg_boolean'] = combined_data['abs_diff_dose_mg'] >= 0.5
    # combined_data['recommended_dose'] = combined_data['recommended_dose_mg'] / combined_data['body_weight']

    # number_of_similar_dose = len(combined_data) - combined_data.diff_dose_mg_boolean.sum()

    # # Keep those with diff dose
    # combined_data = combined_data[combined_data.diff_dose_mg_boolean==True].reset_index(drop=True)

    # Count number of non-therapeutic range
    number_of_non_therapeutic_range = len(combined_data) - combined_data.within_range.sum()

    # # Keep non-therapeutic range only
    # combined_data = combined_data[combined_data.within_range == False].reset_index(drop=True)

    # combined_data['diff_dose'] = combined_data['dose'] - combined_data['recommended_dose']
    # combined_data['abs_diff_dose'] = abs(combined_data['dose'] - combined_data['recommended_dose'])

    combined_data['CURATE_could_be_useful'] = (combined_data.acceptable_deviation==True) & \
    (combined_data.wrong_range==False) & \
        (combined_data.reasonable_dose==True) & \
            (combined_data.within_range==False)

    return combined_data

def group_comparison(file_string):
    """ 
    Use Mann Whitney U test and Spearman's rank correlation coefficient
    to compare between top 2 RW and PPM methods.
    
    Output: printed results of the 2 tests
    """

    dat = read_file_and_remove_unprocessed_pop_tau(file_string)

    # Add type column
    dat['type'] = ""
    for i in range(len(dat)):
        if 'L_' in dat.method[i]:
            dat.loc[i, 'type'] = 'linear'
        else:
            dat.loc[i, 'type'] = 'quadratic'

    dat['approach'] = ""
    dat['origin_inclusion'] = ""
    dat['pop_tau'] = ""
    for i in range(len(dat)):
        if 'Cum' in dat.method[i]:
            dat.loc[i, 'approach']  = 'Cum'
        elif 'PPM' in dat.method[i]:
            dat.loc[i, 'approach'] = 'PPM'
        else: dat.loc[i, 'approach'] = 'RW'

        if 'wo_origin' in dat.method[i]:
            dat.loc[i, 'origin_inclusion'] = 'wo_origin'
        elif 'origin_dp' in dat.method[i]:
            dat.loc[i, 'origin_inclusion'] = 'origin_dp'
        else: dat.loc[i, 'origin_inclusion'] = 'origin_int'

        if 'pop_tau' in dat.method[i]:
            dat.loc[i, 'pop_tau'] = True
        else: dat.loc[i, 'pop_tau'] = False

    PPM_origin_dp = dat[dat.method=='L_PPM_origin_dp']['deviation'].to_numpy()
    PPM_wo_origin = dat[dat.method=='L_PPM_wo_origin']['deviation'].to_numpy()

    res = mannwhitneyu(PPM_origin_dp, PPM_wo_origin)
    print(f'PPM spearman | {stats.spearmanr(PPM_origin_dp, PPM_wo_origin)}')
    print(f'PPM mann-whitney| {mannwhitneyu(PPM_origin_dp, PPM_wo_origin)}')

    RW_origin_int = dat[dat.method=='L_RW_origin_int']['deviation'].to_numpy()
    RW_wo_origin = dat[dat.method=='L_RW_wo_origin']['deviation'].to_numpy()

    res = mannwhitneyu(RW_origin_int, RW_wo_origin)

    print(f'RW spearman | {stats.spearmanr(RW_origin_int, RW_wo_origin)}')
    print(f'RW mann-whitney| {mannwhitneyu(RW_origin_int, RW_wo_origin)}')

    dat = df.copy()
    PPM_origin_dp = dat[dat.method=='L_PPM_origin_dp']['abs_deviation'].to_numpy()
    PPM_wo_origin = dat[dat.method=='L_PPM_wo_origin']['abs_deviation'].to_numpy()

    res = mannwhitneyu(PPM_origin_dp, PPM_wo_origin)
    print(f'PPM spearman | {stats.spearmanr(PPM_origin_dp, PPM_wo_origin)}')
    print(f'PPM mann-whitney| {mannwhitneyu(PPM_origin_dp, PPM_wo_origin)}')

    RW_origin_int = dat[dat.method=='L_RW_origin_int']['abs_deviation'].to_numpy()
    RW_wo_origin = dat[dat.method=='L_RW_wo_origin']['abs_deviation'].to_numpy()

    res = mannwhitneyu(RW_origin_int, RW_wo_origin)

    print(f'RW spearman | {stats.spearmanr(RW_origin_int, RW_wo_origin)}')
    print(f'RW mann-whitney| {mannwhitneyu(RW_origin_int, RW_wo_origin)}')

##### Meeting with NUH ######

def perc_days_within_target_tac(result_df):
    """
    Barplot of percentage of days within target tac range against each patient.
    
    Input: cal_pred - calibration and efficacy-driven dosing data for each prediction day
    
    Output: dat - dataframe for plotting
    """
    # Plot percentage of days within target tac range
    sns.set(font_scale=2, rc={'figure.figsize':(10,10)})
    sns.set_style('whitegrid')

    dat = result_df.copy()

    for i in range(len(dat)):
        if 'L' in dat.loc[i, 'method']:
            dat.loc[i, 'type'] = 'linear'
        else: 
            dat.loc[i, 'type'] = 'quadratic'
    dat = dat[['pred_day', 'patient', 'method', 'type', 'response']]
    dat = dat.reset_index(drop=True)

    dat['within_tac_range'] = (dat['response'] >= 8) & (dat['response'] <= 10)
    dat = (dat.groupby('patient')['within_tac_range'].sum())/ (dat.groupby('patient')['pred_day'].count()) * 100
    dat = dat.to_frame()
    dat.columns = ['perc']
    dat.reset_index(inplace=True)

    p = sns.barplot(data=dat, x='patient', y='perc', palette='Paired')
    p.set_xlabel('Patient')
    p.set_ylabel('Days (%)')
    p.set_title('Days outside target tacrolimus range (%)')
    p.set_ylim([0,100])

    # Shapiro test for percentages
    shapiro_test = stats.shapiro(dat.perc)
    if shapiro_test.pvalue < 0.05:
        print('reject null hypothesis, assume not normal')
    else:
        print('fail to reject null hypothesis, assume normal')

    # Descriptive stats
    print(dat.perc.describe())

def perc_days_outside_target_tac(result_df):
    """
    Barplot of percentage of days outside target tac range against each patient.
    
    Input: cal_pred - calibration and efficacy-driven dosing data for each prediction day
    
    Output: dat - dataframe for plotting
    """
    # Plot percentage of days within target tac range
    sns.set(font_scale=2, rc={'figure.figsize':(10,10)})
    sns.set_style('whitegrid')

    dat = result_df.copy()

    for i in range(len(dat)):
        if 'L' in dat.loc[i, 'method']:
            dat.loc[i, 'type'] = 'linear'
        else: 
            dat.loc[i, 'type'] = 'quadratic'
    dat = dat[['pred_day', 'patient', 'method', 'type', 'response']]
    dat = dat.reset_index(drop=True)

    dat['outside_tac_range'] = (dat['response'] < 8) | (dat['response'] > 10)
    dat = (dat.groupby('patient')['outside_tac_range'].sum())/ (dat.groupby('patient')['pred_day'].count()) * 100
    dat = dat.to_frame()
    dat.columns = ['perc']
    dat.reset_index(inplace=True)

    p = sns.barplot(data=dat, x='patient', y='perc', palette='Paired')
    p.set_xlabel('Patient')
    p.set_ylabel('Days (%)')
    p.set_title('Days outside target tacrolimus range (%)')
    p.set_ylim([0,100])

    # Shapiro test for percentages
    shapiro_test = stats.shapiro(dat.perc)
    if shapiro_test.pvalue < 0.05:
        print('reject null hypothesis, assume not normal')
    else:
        print('fail to reject null hypothesis, assume normal')

    # Descriptive stats
    print(dat.perc.describe())
    
def median_perc_within_acc_dev(result_df):
    """
    Boxplot of median percentage of predictions within acceptable deviation. Conduct Kruskal Wallis and Levene's test.
    
    Input: result_df - results after all methods are applied
    Output: boxplot
    """

    # Find percentage of predictions within acceptable deviation
    dat = result_df[['patient', 'method', 'deviation']]
    dat['acceptable'] = (dat['deviation'] > -3) & (dat['deviation'] < 1)
    dat = (dat.groupby(['method','patient'])['acceptable'].sum())/(dat.groupby(['method','patient'])['acceptable'].count()) * 100
    dat = dat.to_frame()
    dat = dat.reset_index()

    # Run normality test on each method
    method_arr = dat.method.unique()
    for i in method_arr:
        method_dat = dat[dat['method']==i].acceptable

        # Shapiro test
        shapiro_test = stats.shapiro(method_dat)
        # if shapiro_test.pvalue < 0.05:
        #     print('reject null hypothesis, assume not normal')
        # else:
        #     print('fail to reject null hypothesis, assume normal')

    # Add 'approach' column
    for i in range(len(dat)):
        if 'Cum' in dat.loc[i, 'method']:
            dat.loc[i, 'approach'] = 'Cumulative'
        elif 'PPM' in dat.loc[i, 'method']:
            dat.loc[i, 'approach'] = 'PPM'
        else:
            dat.loc[i, 'approach'] = 'RW'

    # Add 'type' column
    for i in range(len(dat)):
        if 'L' in dat.loc[i, 'method']:
            dat.loc[i, 'type'] = 'linear'
        else:
            dat.loc[i, 'type'] = 'quadratic'

    # Add 'origin_inclusion' column
    for i in range(len(dat)):
        if 'wo_origin' in dat.loc[i, 'method']:
            dat.loc[i, 'origin_inclusion'] = 'wo_origin'
        elif 'origin_dp' in dat.loc[i, 'method']:
            dat.loc[i, 'origin_inclusion'] = 'origin_dp'
        else:
            dat.loc[i, 'origin_inclusion'] = 'origin_int'

    fig, ax = plt.subplots(nrows=1, ncols=3)

    # Boxplot by approach
    sns.set_theme(style="whitegrid",font_scale=1.2)
    ax = sns.catplot(data=dat, x='origin_inclusion', y='acceptable', col='approach', hue='type', kind='box')
    ax.set_xlabels('Approach')
    ax.set_ylabels('Within acceptable deviation range (%)')
    ax.fig.subplots_adjust(top=0.8)
    ax.fig.suptitle('Percentage of predictions where deviation is within acceptable range (%)')
    
    print(dat.groupby(['type','approach','origin_inclusion'])['patient'].count())
    

    # # Boxplot by type
    # sns.set_theme(style="whitegrid",font_scale=1.2)
    # ax = sns.catplot(data=dat, x='approach', y='acceptable', col='type', hue='origin_inclusion', kind='box')
    # ax.set_xlabels('Approach')
    # ax.set_ylabels('Within acceptable deviation range (%)')
    # ax.fig.subplots_adjust(top=0.8)
    # ax.fig.suptitle('Percentage of predictions where deviation is within acceptable range (%)')
    
    # # Boxplot for Top 3 highest medians
    # top_3 = dat.groupby('method')['acceptable'].median().sort_values(ascending=False).iloc[0:6].to_frame()
    # top_3 = top_3.reset_index().method.unique()
    # list_of_top_3 = top_3.tolist()
    # top_3 = dat[dat['method'].isin(list_of_top_3)]
    # top_3.method = top_3.method.astype("category")
    # top_3.method.cat.set_categories(list_of_top_3, inplace=True)
    # top_3 = top_3.sort_values(['method'])

    # sns.set_theme(font_scale=2)
    # sns.set_style('whitegrid')
    # ax = sns.boxplot(x='method', y='acceptable', data=top_3)
    # ax.set_xticklabels(ax.get_xticklabels(),rotation = 30)
    # ax.set_xlabel('Method')
    # ax.set_ylabel('Within acceptable deviation range (%)')
    # ax.set_title('Top 3 Medians of Percentage of Predictions within Acceptable Deviation Range')
    
    # Run kruskal wallis test on each method
    method_arr = dat.method.unique()
    j = 0
    for i in method_arr:
        method_dat[j] = dat[dat['method']==i].acceptable
        j = j + 1

    # Kruskal wallis test for equal medians
    stats.kruskal(method_dat[0], method_dat[1], method_dat[2], method_dat[3], method_dat[4], method_dat[5],
                  method_dat[6], method_dat[7], method_dat[8], method_dat[9], method_dat[10], method_dat[11],
                  method_dat[12], method_dat[13], method_dat[14], method_dat[15], method_dat[16], method_dat[17])

    # Levene test for equal variances
    from scipy.stats import levene
    stat, p = levene(method_dat[0], method_dat[1], method_dat[2], method_dat[3], method_dat[4], method_dat[5],
                  method_dat[6], method_dat[7], method_dat[8], method_dat[9], method_dat[10], method_dat[11],
                  method_dat[12], method_dat[13], method_dat[14], method_dat[15], method_dat[16], method_dat[17])

def can_benefit(result_df):
    """
    Interpolate to find percentage of possible dosing events for when prediction and observed response are outside range.
    Find percentage of dosing events that our model can potentially outperform SOC (when both observed and predicted values are outside range, with
    prediction within acceptable deviation). 
    Create boxplot of dosing events (%) against method.
    
    Input:
    result_df
    """
    # Interpolate to find percentage of possible dosing events for when prediction and observed response are outside range
    dat = result_df[['patient', 'method', 'pred_day', 'dose', 'response', 'coeff_2x', 'coeff_1x', 'coeff_0x', 'prediction', 'deviation']]

    # for i in range(len(dat)):
    #     # Create function
    #     coeff = dat.loc[i, 'coeff_2x':'coeff_0x'].apply(float).to_numpy()
    #     coeff = coeff[~np.isnan(coeff)]
    #     p = np.poly1d(coeff)
    #     x = np.linspace(0, max(dat.dose)+ 2)
    #     y = p(x)
    #     order = y.argsort()
    #     y = y[order]
    #     x = x[order]

    #     dat.loc[i, 'interpolated_dose_8'] = np.interp(8, y, x)
    #     dat.loc[i, 'interpolated_dose_9'] = np.interp(9, y, x)
    #     dat.loc[i, 'interpolated_dose_10'] = np.interp(10, y, x)

    # dat[['interpolated_dose_8','interpolated_dose_9','interpolated_dose_10']].describe() # Minimum 0mg, all are possible dosing events

    # Find percentage of predictions where both observed and prediction response are outside range
    for i in range(len(dat)):
        dat.loc[i, 'both_outside'] = False
        if (dat.loc[i, 'prediction'] > 10) or (dat.loc[i, 'prediction'] < 8):
            if (dat.loc[i, 'prediction'] > 10) or (dat.loc[i, 'prediction'] < 8):
                dat.loc[i, 'both_outside'] = True

    dat['acceptable_deviation'] = (dat['deviation'] > -3) & (dat['deviation'] < 1)

    dat['can_benefit'] = dat['acceptable_deviation'] & dat['both_outside']

    # If can correctly identify out of range, with acceptable deviation, can benefit
    dat = (dat.groupby(['method', 'patient'])['can_benefit'].sum()) / (dat.groupby(['method', 'patient'])['can_benefit'].count()) * 100
    dat = dat.to_frame().reset_index()

    # Shapiro test
    # Normality test (result: assume non-normal)
    method_arr = dat.method.unique()
    method_dat = {}
    j = 0
    for i in method_arr: 
        method_dat[j] = dat[dat['method']==i].can_benefit
        shapiro_test = stats.shapiro(method_dat[j])
        print(shapiro_test.pvalue < 0.05)
        j = j + 1

    ax = sns.boxplot(data=dat, x='method', y='can_benefit', dodge=False)
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
    ax.set_xlabel(None)
    ax.set_ylabel('Dosing events (%)')
    ax.set_title('Dosing events that can potentially outperform SOC with CURATE (%)')
    plt.legend(loc='upper right', bbox_to_anchor=(1,1))

    dat.can_benefit.describe() # Shapiro test reject null hypo, assume non-normal

def modified_TTR(result_df):
    """
    Calculate CURATE modified TTR and physician modified TTR.
    Create barplot of modified TTR vs method grouped under 'physician' and 'CURATE'
    
    Input: 
    result_df - dataframe of results of applied methods.
    """
    sns.set(font_scale=2, rc={'figure.figsize':(15,10)})
    sns.set_style('whitegrid')

    # Find percentage of success instances
    dat = result_df[['patient', 'method', 'response', 'prediction']]

    # CURATE success
    for i in range(len(dat)):
        dat.loc[i, 'success'] = False
        if (dat.loc[i, 'response'] < 10) and (dat.loc[i, 'response'] > 8): # if both within range
            if (dat.loc[i, 'prediction'] < 10) and (dat.loc[i, 'prediction'] > 8):
                dat.loc[i, 'success'] = True
        elif (dat.loc[i, 'response'] >= 10) or (dat.loc[i, 'response'] <= 8): # if both outside range
            if (dat.loc[i, 'prediction'] >= 10) or (dat.loc[i, 'prediction'] <= 8):
                dat.loc[i, 'success'] = True

    CURATE_TTR = (dat.groupby(['method','patient'])['success'].sum())/(dat.groupby(['method','patient'])['success'].count()) * 100

    # Normality test (result: assume normal)
    CURATE_TTR = CURATE_TTR.to_frame().reset_index()
    method_arr = CURATE_TTR.method.unique()
    method_dat = {}
    j = 0
    for i in method_arr:
        method_dat[j] = CURATE_TTR[CURATE_TTR['method']==i].success
        shapiro_test = stats.shapiro(method_dat[j])
        # print(shapiro_test.pvalue)
        j = j + 1
    CURATE_TTR['source'] = 'CURATE'

    # Physician success
    phys_TTR = dat[(dat['method']=='L_Cum_wo_origin') | (dat['method']=='Q_Cum_wo_origin')]
    phys_TTR['success'] = (phys_TTR['response'] < 10) & (phys_TTR['response'] > 8)
    phys_TTR = (phys_TTR.groupby(['method', 'patient'])['success'].sum())/(phys_TTR.groupby(['method', 'patient'])['success'].count()) * 100
    phys_TTR = phys_TTR.to_frame().reset_index()
    for i in range(len(phys_TTR)):
        if 'L' in phys_TTR.loc[i, 'method']:
            phys_TTR.loc[i, 'method'] = 'linear'
        else:
            phys_TTR.loc[i, 'method'] = 'quadratic'
    phys_TTR['source'] = 'Physician'

    # Normality test (result: assume normal)
    method_arr = phys_TTR.method.unique()
    method_dat = {}
    j = 0
    for i in method_arr: 
        method_dat[j] = phys_TTR[phys_TTR['method']==i].success
        # shapiro_test = stats.shapiro(method_dat[j])
        print(shapiro_test.pvalue < 0.05)
        j = j + 1

    dat = pd.concat([CURATE_TTR, phys_TTR])

    ax = sns.barplot(data=dat, x='method', y='success', hue='source', ci='sd', capsize=.2, dodge=False)
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
    ax.set_xlabel(None)
    ax.set_ylabel('Modified TTR (%)')
    ax.set_title('Modified TTR for Physician vs CURATE (%)')
    plt.legend(loc='upper right', bbox_to_anchor=(1,1))

    return dat

def wrong_range(result_df):
    """
    Find percentage of dosing events when model predicted wrong range. 
    Find percentage of dosing events when SOC is outside target range.
    Conduct Shapiro test.
    Boxplot of percentage of wrong range against each method. 
    
    Input: result_df - results after all methods are applied
    Output: boxplot
    """
    
    # Find dosing events when model predicted the wrong range
    dat = result_df[['patient', 'method', 'prediction', 'response']]

    # Create boolean, true when model predict wrong range
    for i in range(len(dat)):
        # All False
        dat.loc[i, 'wrong_range'] = False
        # Unless condition 1: prediction within range, response outside range
        if (dat.loc[i, 'prediction'] >= 8) and (dat.loc[i, 'prediction'] <= 10):
            if (dat.loc[i, 'response'] > 10) or (dat.loc[i, 'response'] < 8):
                dat.loc[i, 'wrong_range'] = True
        # Unless condition 2: prediction outside range, response within range
        elif (dat.loc[i, 'prediction'] > 10) or (dat.loc[i, 'prediction'] < 8):
            if (dat.loc[i, 'response'] >= 8) and (dat.loc[i, 'response'] <= 10):
                dat.loc[i, 'wrong_range'] = True

    dat = (dat.groupby(['method', 'patient'])['wrong_range'].sum()) / (dat.groupby(['method', 'patient'])['wrong_range'].count()) * 100
    dat = dat.to_frame().reset_index()
    dat['source'] = 'CURATE'

    # Create another dataframe
    dat_physician = result_df[['patient', 'method', 'prediction', 'response']]
    dat_physician = dat_physician[(dat_physician['method']=='L_Cum_wo_origin') | (dat_physician['method']=='Q_Cum_wo_origin')]
    dat_physician = dat_physician.reset_index(drop=True)

    # Create boolean, true if response is outside range
    for i in range(len(dat_physician)):
        # Set boolean default as false
        dat_physician.loc[i, 'wrong_range'] = False
        # Create boolean as True if outside range
        if (dat_physician.loc[i, 'response'] > 10) or (dat_physician.loc[i, 'response'] < 8):
            dat_physician.loc[i, 'wrong_range'] = True

    dat_physician.groupby(['method', 'patient'])['wrong_range'].count()
    dat_physician = (dat_physician.groupby(['method', 'patient'])['wrong_range'].sum()) / (dat_physician.groupby(['method', 'patient'])['wrong_range'].count()) * 100
    dat_physician = dat_physician.to_frame().reset_index()
    dat_physician['source'] = 'Physician'

    # Rename methods to linear and quadratic only
    for i in range(len(dat_physician)):
        if 'L' in dat_physician.loc[i, 'method']:
            dat_physician.loc[i, 'method'] = 'linear'
        else:
            dat_physician.loc[i, 'method'] = 'quadratic'

    dat = pd.concat([dat, dat_physician])

    # Shapiro test
    # Normality test (result: reject, assume non-normal)
    method_arr = dat.method.unique()
    method_dat = {}
    j = 0
    for i in method_arr: 
        method_dat[j] = dat[dat['method']==i].wrong_range
        shapiro_test = stats.shapiro(method_dat[j])
        # print(shapiro_test.pvalue < 0.05)
        j = j + 1

    # Boxplot
    # sns.set(font_scale=2, rc={'figure.figsize':(15,10)})
    sns.set_theme(font_scale=2)
    sns.set_style('whitegrid')
    ax = sns.boxplot(data=dat, x='method', y='wrong_range', hue='source', dodge=False)
    ax.set_xticklabels(ax.get_xticklabels(),rotation = 90)
    ax.set_xlabel(None)
    ax.set_ylabel('Wrong Range Predicted (%)')
    ax.set_title('Wrong Range Predicted  (%)')
    plt.legend(loc='upper right', bbox_to_anchor=(1.25,1))

    return dat