def case_series_120_day_4_regression():
    """Scatter and line plot of day 4 recommendation for patient 120"""
    dat_original, combined_df = case_series_120()

    # Subset first prediction since it could outperform SOC
    dat = combined_df[combined_df.pred_day==4].reset_index(drop=True)

    sns.set(style='white', font_scale=2,
           rc={"figure.figsize":(7,7), "xtick.bottom":True, "ytick.left":True})

    # Plot regression line
    x = np.array([dat.x[0],dat.x[1]])
    y = np.array([dat.y[0],dat.y[1]])
    a, b = np.polyfit(x, y, 1)
    x_values = np.linspace(0, 3)
    plt.plot(x_values, a*x_values + b, linestyle='-', color='y')

    # Plot scatter points
    plt.scatter(x, y, s=100, color='y')

    # Plot therapeutic range
    plt.axhspan(8, 10, facecolor='grey', alpha=0.2)

    # Label days
    for i in range(dat.shape[0]):
        plt.text(x=dat.x[i]+0.1,y=dat.y[i]+0.1,s=int(dat.day[i]),
                 fontdict=dict(color='black',size=13),
                 bbox=dict(facecolor='y', ec='black', alpha=0.5, boxstyle='circle'))

    sns.despine()
    plt.title('Day 4 recommendation')
    plt.xlabel('Dose (mg)')
    plt.ylabel('Tacrolimus level (ng/ml)')
    plt.xticks(np.arange(0,3.5,step=0.5))
    plt.xlim(0,2.5)

    plt.savefig('patient_120_case_series_reco.png',dpi=500)
    
    return dat

def values_by_dosing_strategy():
    """
    Find therapeutic range values by dosing strategy

    Output: 
    - Printed values of number of patients, first day 
    to achieve therapeutic range, and percentage of days in
    therapeutic range, by dosing strategy
    - combined_df: final dataframe of all values by dosing
    strategy
    """

    # Import data and list of patients
    df = import_raw_data_including_non_ideal()
    list_of_patients = find_list_of_patients()

    # Create therapeutic range column
    df['therapeutic_range'] = ""
    for i in range(len(df)):
        if (df.response[i] >= therapeutic_range_lower_limit) & (df.response[i] <= therapeutic_range_upper_limit):
            df.loc[i, 'therapeutic_range'] = 'therapeutic'
        else:
            df.loc[i, 'therapeutic_range'] = 'non-therapeutic'

    # Find percentage of therapeutic range per patient
    perc_therapeutic_range = df.groupby('patient')['therapeutic_range'].apply(lambda x: (x=='therapeutic').sum()/x.count()*100)
    perc_therapeutic_range = perc_therapeutic_range.to_frame().reset_index()
    perc_therapeutic_range = perc_therapeutic_range.rename(columns={'therapeutic_range':'perc_therapeutic_range'})

    # Find first day of achieving therapeutic range per patient
    first_day_of_therapeutic_range = df[df.therapeutic_range=='therapeutic'].groupby('patient')['day'].first()
    first_day_of_therapeutic_range = first_day_of_therapeutic_range.to_frame().reset_index()
    first_day_of_therapeutic_range = first_day_of_therapeutic_range.rename(columns={'day':'first_day_therapeutic_range'})

    # Categorise patients
    dosing_strategy = df.groupby('patient')['dose'].apply(lambda x: \
                                                          'distributed' if ((x.max()-x.min()) >= dosing_strategy_cutoff) \
                                                          else 'repeated')
    dosing_strategy = dosing_strategy.to_frame().reset_index()
    dosing_strategy = dosing_strategy.rename(columns={'dose':'dosing_strategy'})

    # Merge all three dataframes
    combined_df = dosing_strategy.merge(first_day_of_therapeutic_range, how='left', on='patient')
    combined_df = combined_df.merge(perc_therapeutic_range, how='left', on='patient')

    print(f'Repeated:\nN = {combined_df.dosing_strategy.value_counts().loc["repeated"]}\n\
    First day to achieve therapeutic range: \
    {combined_df[combined_df.dosing_strategy=="repeated"]["first_day_therapeutic_range"].describe().loc["50%"]} [IQR \
    {combined_df[combined_df.dosing_strategy=="repeated"]["first_day_therapeutic_range"].describe().loc["25%"]} - \
    {combined_df[combined_df.dosing_strategy=="repeated"]["first_day_therapeutic_range"].describe().loc["75%"]}] days\n\
    Days in therapeutic range (%): \
    {combined_df[combined_df.dosing_strategy=="repeated"]["perc_therapeutic_range"].describe().loc["50%"]:.2f} [IQR \
    {combined_df[combined_df.dosing_strategy=="repeated"]["perc_therapeutic_range"].describe().loc["25%"]:.2f} - \
    {combined_df[combined_df.dosing_strategy=="repeated"]["perc_therapeutic_range"].describe().loc["75%"]:.2f}]')

    print(f'\nDistributed:\nN = {combined_df.dosing_strategy.value_counts().loc["distributed"]}\n\
    First day to achieve therapeutic range: \
    {combined_df[combined_df.dosing_strategy=="distributed"]["first_day_therapeutic_range"].describe().loc["50%"]} [IQR \
    {combined_df[combined_df.dosing_strategy=="distributed"]["first_day_therapeutic_range"].describe().loc["25%"]} - \
    {combined_df[combined_df.dosing_strategy=="distributed"]["first_day_therapeutic_range"].describe().loc["75%"]}] days\n\
    Days in therapeutic range (%): \
    {combined_df[combined_df.dosing_strategy=="distributed"]["perc_therapeutic_range"].describe().loc["50%"]:.2f} [IQR \
    {combined_df[combined_df.dosing_strategy=="distributed"]["perc_therapeutic_range"].describe().loc["25%"]:.2f} - \
    {combined_df[combined_df.dosing_strategy=="distributed"]["perc_therapeutic_range"].describe().loc["75%"]:.2f}]')

    return combined_df

def case_series_120(plot=False):
    """
    Line and scatter plot of response vs dose, for each day and
    with regression line for RW by prediction day
    """

    dat = pd.read_excel(result_file, sheet_name='result')

    # Subset L_RW_wo_origin and patient 118
    dat = dat[(dat.method=='L_RW_wo_origin') &  (dat.patient==120)]

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

    if plot==True:
        # Plot
        sns.set(font_scale=1.2, rc={"figure.figsize": (16,10), "xtick.bottom":True, "ytick.left":True},
                style='white')
        g = sns.lmplot(data=combined_df, x='x', y='y', hue='pred_day', ci=None, legend=False)

        # Make unused data points have lighter edge colors
        ec = colors.to_rgba('black')
        ec = ec[:-1] + (0.3,)

        # Add labels for numbered circles representing day from which dose-response pair 
        # was obtained from
        for i in range(combined_df.shape[0]):
            plt.text(x=combined_df.x[i]+0.2,y=combined_df.y[i]+0.2,s=int(combined_df.day[i]), 
            fontdict=dict(color='black',size=13),
            bbox=dict(facecolor='white', ec='black', alpha=0.5, boxstyle='circle'))

            plt.text(x=4.5+0.3,y=12+0.3,s=23, 
            fontdict=dict(color='black',size=13),
            bbox=dict(facecolor='white', ec=ec, boxstyle='circle'))

            plt.text(x=4+0.3,y=10+0.3,s=24, 
            fontdict=dict(color='black',size=13),
            bbox=dict(facecolor='white', ec=ec, boxstyle='circle'))
            
            plt.text(x=4+0.3,y=11.7+0.3,s=25, 
            fontdict=dict(color='black',size=13),
            bbox=dict(facecolor='white', ec=ec, boxstyle='circle'))

            plt.text(x=4+0.3,y=11.3+0.3,s=26, 
            fontdict=dict(color='black',size=13),
            bbox=dict(facecolor='white', ec=ec, boxstyle='circle'))
            
            plt.text(x=3.5+0.3,y=10.6+0.3,s=27, 
            fontdict=dict(color='black',size=13),
            bbox=dict(facecolor='white', ec=ec, boxstyle='circle'))

        plt.legend(bbox_to_anchor=(1.06,0.5), loc='center left', title='Day of Prediction', frameon=False)
        plt.xlabel('Tacrolimus dose (mg)')
        plt.ylabel('Tacrolimus level (ng/ml)')
        plt.axhspan(8, 10, facecolor='grey', alpha=0.2) # Shaded region for therapeutic range

        # Add data point of day 23 to 27
        plt.plot(4.5, 12, marker="o", markeredgecolor="black", markerfacecolor="white")
        plt.plot(4, 10, marker="o", markeredgecolor="black", markerfacecolor="white")
        plt.plot(4, 11.7, marker="o", markeredgecolor="black", markerfacecolor="white")
        plt.plot(4, 11.3, marker="o", markeredgecolor="black", markerfacecolor="white")
        plt.plot(3.5, 10.6, marker="o", markeredgecolor="black", markerfacecolor="white")
        plt.savefig('patient_120_RW_profiles.png', dpi=500, facecolor='w', bbox_inches='tight')

    return dat_original, combined_df

def case_series_120_response_vs_day():
    """
    Scatter and line plot of patient 120 of response vs day,
    with first day to achieve therapeutic range, for SOC and CURATE.AI-assisted dosing
    """
    # clean_dat = pd.read_excel('output (with pop tau by LOOCV).xlsx', sheet_name='clean')
    clean_dat = pd.read_excel('Retrospective Liver Transplant Data_simplified.xlsx', sheet_name='120')
    clean_dat = clean_dat.rename(columns={'Tac level (prior to am dose)':'response', 'Eff 24h Tac Dose':'dose', 'Day #':'day'})
    clean_dat['response'] = clean_dat['response'].shift(-1)
    clean_dat = clean_dat[clean_dat['response'].notna()]

    # Shift dose one unit up, fill NaN with 0mg, reset index
    clean_dat['dose'] = clean_dat['dose'][:-2]
    clean_dat['dose'] = clean_dat['dose'].fillna('0mg')
    clean_dat = clean_dat.reset_index(drop=True)

    # Remove 'mg'
    for i in range(len(clean_dat)):
        size = len(clean_dat.dose[i])
        clean_dat.dose[i] = clean_dat.dose[i][:size - 2]

    dat_original, combined_df = case_series_120()

    # Subset patient 120
    # clean_dat = clean_dat[(clean_dat.patient==120) & (clean_dat.day >= 2) & (clean_dat.day <= 4)].reset_index(drop=True)
    # clean_dat = clean_dat[(clean_dat.patient==120)].reset_index(drop=True)

    predicted_response = (dat_original.loc[0, 'coeff_1x'] * 2) + (dat_original.loc[0, 'coeff_0x'])
    predicted_dose = 2

    # Plot
    fig, axes = plt.subplots(figsize=(7,7))
    sns.set(style='white', font_scale=2,
           rc={"xtick.bottom":True, "ytick.left":True})

    plt.plot(clean_dat.day, clean_dat.response, 'yo', linestyle='-', ms=10)
    plt.scatter(x=clean_dat.day[0], y=clean_dat.response[0], color='y', s=100, label='Standard of care dosing')
    plt.plot(4, predicted_response, 'm^', ms=10, label='First day of therapeutic range\nwith CURATE.AI-assisted dosing')
    plt.plot(8, 9.9, 'go', ms=10, label='First day of therapeutic range\nwith standard of care dosing')
    # plt.vlines(4, ymin=0, ymax=predicted_response, linestyle='--', color='m')
    # plt.vlines(8, ymin=0, ymax=9.9, linestyle='--', color='g')
    plt.ylim(0,max(clean_dat.response+1))

    sns.despine()
    plt.xticks(np.arange(2,max(clean_dat.day),step=4))
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

    plt.tight_layout()
    plt.savefig('patient_120_first_day.png',dpi=500, bbox_inches='tight')
    
    return clean_dat

def ks_test_result(df, metric_string):
    ks_test = stats.kstest(df, "norm").pvalue
    
    if ks_test < 0.05:
        result_string = 'reject normality'
    else:
        result_string = 'assume normality'

    print(f'{metric_string}:\nKS test p-value = {ks_test:.2f}, {result_string}')
    
    return ks_test
    
def clinically_relevant_performance_metrics(result_file=result_file_total, all_data_file=all_data_file_total, dose='total'):
    """Clinically relevant performance metrics. 
    Calculate the results, conduct statistical tests, and
    print them out. 
    
    Instructions: Uncomment first block of code to write output to txt file.
    """
    # Uncomment to write output to txt file
    # file_path = 'Clinically relevant performance metrics.txt'
    # sys.stdout = open(file_path, "w")
    if dose == 'total':
        result_file = result_file_total
        all_data_file = all_data_file_total
    else:
        result_file = result_file_evening
        all_data_file = all_data_file_evening

    original_stdout = sys.stdout
    with open('clinically_relevant_perf_metrics_'+ dose +'.txt', 'w') as f:
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

    return acceptable_CURATE, acceptable_SOC, unacceptable_overprediction, unacceptable_underprediction
