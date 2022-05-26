import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt
import pandas as pd

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