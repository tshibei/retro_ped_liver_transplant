import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error

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

def prediction_error(file_string):
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

def RMSE_plot(file_string):
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

    g = sns.catplot(data=dat, x='OG_method', y='rmse', hue='pop_tau', kind='bar', col='approach', sharex=False)
    g.set_xticklabels(rotation=90)
    g.set_ylabels('RMSE')
    plt.savefig('RMSE_by_approach.png', bbox_inches='tight', dpi=300, facecolor='w')

    return dat

def RMSE_method(dat):
    """Find RMSE by method"""
    rmse = mean_squared_error(dat.response, dat.prediction, squared=False)
    return pd.Series(dict(rmse=rmse))

def ideal_over_under_pred(file_string):
    """Bar plot of percentage of ideal/over/under predictions, by method and pop tau"""
    dat = read_file_and_remove_unprocessed_pop_tau(file_string)

    # Calculate % of predictions within acceptable error, overprediction, and underprediction
    ideal = dat.groupby('method')['deviation'].apply(lambda x: ((x > -2) & (x < 1.5)).sum()/ x.count()*100).reset_index()
    ideal['result'] = 'ideal'
    over = dat.groupby('method')['deviation'].apply(lambda x: ((x < -2)).sum()/ x.count()*100).reset_index()
    over['result'] = 'over'
    under = dat.groupby('method')['deviation'].apply(lambda x: ((x > 1.5)).sum()/ x.count()*100).reset_index()
    under['result'] = 'under'

    # Combine results into a dataframe
    metric_df = pd.concat([ideal, over, under]).reset_index(drop=True)

    # Add pop tau column, and remove 'pop_tau' from method
    metric_df['pop_tau'] = ""
    for i in range(len(metric_df)):
        if 'pop_tau' in metric_df.method[i]:
            metric_df.loc[i, 'pop_tau'] = 'pop tau'
            metric_df.loc[i, 'method'] = metric_df.loc[i, 'method'][:-8]
        else: 
            metric_df.loc[i, 'pop_tau'] = 'no pop tau'

    # # Perform shapiro test (result: some pvalue < 0.05, some > 0.05)
    # kstest_result = metric_df.groupby(['pop_tau', 'result'])['deviation'].apply(lambda x: stats.shapiro(x).pvalue < 0.05).reset_index()

    # # Describe ideal/over/under prediction results
    # pd.set_option('display.float_format', lambda x: '%.2f' % x)
    # metric_df.groupby(['pop_tau', 'result'])['deviation'].describe()

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
    metric_df.columns = ['method', 'perc_predictions', 'result', 'pop_tau']

    return metric_df

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

def indiv_profiles(file_string):
    """Scatter plot of inidividual profiles, longitudinally, and response vs dose"""

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

    # dat
    # # dat
    sns.set(font_scale=1.2)
    sns.set_style('white')

    g = sns.relplot(data=dat, x='day', y='response', hue='within_range', col='patient', col_wrap=4, style='dose_range',
               height=1.5, aspect=1)

    g.map(plt.axhline, y=10, ls='--', c='black')
    g.map(plt.axhline, y=8, ls='--', c='black')

    plt.savefig('indiv_pt_profile_by_day.png', dpi=500, facecolor='w', bbox_inches='tight')

    sns.set(font_scale=1.2)
    sns.set_style('white')

    g = sns.relplot(data=dat, x='dose', y='response', hue='day', col='patient', col_wrap=4, style='dose_range',
               height=1.5, aspect=1)

    g.map(plt.axhline, y=10, ls='--', c='black')
    g.map(plt.axhline, y=8, ls='--', c='black')

    plt.savefig('indiv_pt_profile_by_dose.png', dpi=500, facecolor='w', bbox_inches='tight')
    
    return dat

def read_file_and_remove_unprocessed_pop_tau(file_string='GOOD OUTPUT DATA\output (with pop tau by LOOCV).xlsx', sheet_string='result'):
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

# LOOCV for all methods

def LOOCV_all_methods(file_string):
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
    with pd.ExcelWriter('all_methods_LOOCV.xlsx') as writer:
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