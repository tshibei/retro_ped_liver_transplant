from plotting import *
from scipy.stats import mannwhitneyu

def CURATE_could_be_useful(file_string='output (with pop tau by LOOCV).xlsx'):
    """
    Exclude cases where CURATE cannot be useful for top 2 methods (PPM and RW), and
    keep those that are useful.
    
    Output: Dataframe describing results
    """
    dat = read_file_and_remove_unprocessed_pop_tau(file_string)
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

    dat['acceptable_deviation'] = (round(dat['deviation'],2) > -2) & (round(dat['deviation'],2) < 1.5)

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

    dat.groupby('method')['diff_dose'].describe().T.applymap('{:,.2f}'.format)

    return dat

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