import seaborn as sns
from scipy import stats
import matplotlib.pyplot as plt

def perc_days_within_target_tac(cal_pred):
    """
    Barplot of percentage of days within target tac range against each patient.
    
    Input: cal_pred - calibration and efficacy-driven dosing data for each prediction day
    
    Output: dat - dataframe for plotting
    """
    # Plot percentage of days within target tac range
    sns.set(font_scale=2, rc={'figure.figsize':(10,10)})
    sns.set_style('whitegrid')

    dat = cal_pred[cal_pred['type']=='quadratic']
    dat = dat.reset_index(drop=True)
    dat['within_tac_range'] = (dat['response'] >= 8) & (dat['response'] <= 10)
    dat = (dat.groupby('patient')['within_tac_range'].sum())/ (dat.groupby('patient')['day'].count()) * 100
    dat = dat.to_frame()
    dat.columns = ['perc']
    dat.reset_index(inplace=True)

    p = sns.barplot(data=dat, x='patient', y='perc', palette='Paired')
    p.set_xlabel('Patient')
    p.set_ylabel('Days (%)')
    p.set_title('Days within target tacrolimus range (%)')
    p.set_ylim([0,95])

    # Shapiro test for percentages
    shapiro_test = stats.shapiro(dat.perc)
    if shapiro_test.pvalue < 0.05:
        print('reject null hypothesis, assume not normal')
    else:
        print('fail to reject null hypothesis, assume normal')

    # Descriptive stats
    dat.perc.describe()
    
    return None

def perc_days_outside_target_tac(cal_pred):
    """
    Barplot of percentage of days outside target tac range against each patient.
    
    Input: cal_pred - calibration and efficacy-driven dosing data for each prediction day
    
    Output: dat - dataframe for plotting
    """
    # Plot percentage of days outside target tac range
    sns.set(font_scale=2, rc={'figure.figsize':(10,10)})
    sns.set_style('whitegrid')

    dat = cal_pred[cal_pred['type']=='quadratic']
    dat = dat.reset_index(drop=True)
    dat['outside_tac_range'] = (dat['response'] < 8) | (dat['response'] > 10)
    dat = (dat.groupby('patient')['outside_tac_range'].sum())/ (dat.groupby('patient')['day'].count()) * 100
    dat = dat.to_frame()
    dat.columns = ['perc']
    dat.reset_index(inplace=True)

    p = sns.barplot(data=dat, x='patient', y='perc', palette='Paired')
    p.set_xlabel('Patient')
    p.set_ylabel('Days (%)')
    p.set_title('Days outside target tacrolimus range (%)')
    p.set_ylim([0,95])

    # Shapiro test for percentages
    shapiro_test = stats.shapiro(dat.perc)
    if shapiro_test.pvalue < 0.05:
        print('reject null hypothesis, assume not normal')
    else:
        print('fail to reject null hypothesis, assume normal')

    # Descriptive stats
    dat.perc.describe()

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
