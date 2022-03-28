import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from scipy import stats
import seaborn as sns
from functools import reduce
pd.options.mode.chained_assignment = None 
from statistics import mean

# Data cleaning

def read_indiv_patient_data(input_file, patient_name, rows_to_skip):
    """Read in individual patient data into a data frame, shift tac level one cell up, 
    remove "mg" from numbers"""

    # Read patient 120 into dataframe
    df = pd.read_excel(input_file, sheet_name=patient_name, skiprows = rows_to_skip)
    df = df[["Day #", "Tac level (prior to am dose)", "Eff 24h Tac Dose"]]

    # Shift tac level one cell up so that tac level is the output to the input of eff 24h dose
    df['Tac level (prior to am dose)'] = df['Tac level (prior to am dose)'].shift(-1)

    # Remove "mg" from eff 24h tac dose
    df['Eff 24h Tac Dose'] = df['Eff 24h Tac Dose'].astype(str).str.replace('mg', '')
    df['Eff 24h Tac Dose'] = df['Eff 24h Tac Dose'].astype(float)

    return df

# Data selection

def keep_longest_chunk(df):
    """
    Find longest chunk of consecutive non-NA data per patient.

    Input: Dataframe of individual patient
    Output: dataframe with the longest chunk of consecutive non-NA data.
    """

    # Create copy of df to manipulate and find largest consec non-NA chunk
    df_temp = df.copy()

    # Create boolean column of non-NA
    df_temp['tac_level_nan'] = (df_temp["Tac level (prior to am dose)"].isna()) 

    # Create index column
    df_temp.reset_index(inplace=True) 

    # Find cumulative sum of non-NA for each index row
    df_cum_sum_non_NA = df_temp['tac_level_nan'].cumsum() 

    # Find number of consecutive non-NA
    df_temp = df_temp.groupby(df_cum_sum_non_NA).agg({'index': ['count', 'min', 'max']})

    # Groupby created useless column level for index, drop it
    df_temp.columns = df_temp.columns.droplevel()

    # Find largest chunk with consec non-NA
    df_temp = df_temp[df_temp['count']==df_temp['count'].max()] 
    df_temp.reset_index(inplace=True)
    
    # Find index of largest chunk to keep in dataframe
    if len(df_temp) > 1: # If there are >1 large chunks with longest length, an error will be printed
        df = print("there are >1 chunks of data with the longest length.")
    else:
        # Find index of largest chunk to keep in dataframe
        min_idx = df_temp.loc[0, 'min'] + 1 # First index where non-NA begins is 1 after the min index, thus add 1 to min index
        max_idx = df_temp.loc[0, 'max'] # Get max index where non-NA chunk ends

        # Keep largest chunk in dataframe
        df = df.iloc[min_idx:max_idx + 1, :] 
    
    return df

# Tests

def normality_test(df):
    """
    Perform both numerical and graphical normality tests.

    Input: Dataframe
    Output: p value of shapiro test, decision, and graphs.
    """

    # Check normality of dataset with shapiro test
    shapiro_test = stats.shapiro(df["Tac level (prior to am dose)"])
    print("Shapiro_test p value:", shapiro_test.pvalue)
    if shapiro_test.pvalue < 0.05:
        print("Thus, reject normality\n")
    else: 
        print("Thus, assume normality\n")

    # Check normality of dataset with probability plot and histogram 
    ## Probability plot
    df["Tac level (prior to am dose)"] = df["Tac level (prior to am dose)"].astype(float)
    stats.probplot(df["Tac level (prior to am dose)"], dist = "norm", plot = pylab)
    pylab.show()

    ## Histogram
    ax = sns.histplot(df["Tac level (prior to am dose)"])

def python_vs_excel_polyfit():
    """ Visualise the difference between polyfit in python vs excel """
    model_python = np.poly1d([0.90768,1.000041, -4.069297])
    model_excel = np.poly1d([1.06, 0, -2.45])
    polyline = np.linspace(2,5,50)
    plt.scatter([3, 3.5, 3.5], [7.1, 11.2, 9.9])
    plt.plot(polyline, model_python(polyline))
    plt.plot(polyline, model_excel(polyline))
    plt.show()

# Methods

def Q_Cum(df):
    """
    Use Q_Cum method to generate predictions and calculate deviations.
    Input: Individual patient data
    Output: Q_Cum results
    """
    # Create dataframe for Q-Cum
    column_names = ['prediction day', 'a', 'b', 'c', 'prediction', 'deviation', 'abs deviation']
    df_Q_Cum = pd.DataFrame(columns = column_names)

    for day_num in range(3, len(df)): # Prediction starts at index 3, after taking 3 sets of unique data points
        pred_day = int(df["Day #"][day_num])

        # Find coefficients of quadratic fit
        fittedParameters = (np.polyfit(df["Eff 24h Tac Dose"][0:day_num], df["Tac level (prior to am dose)"][0:day_num], 2))

        # Calculate prediction based on quad fit
        prediction = np.polyval(fittedParameters, df["Eff 24h Tac Dose"][day_num])

        # Calculate deviation from prediction
        deviation = prediction - df["Tac level (prior to am dose)"][day_num]
        abs_deviation = abs(deviation)

        # Add the prediction day, coefficients, prediction, and deviation below dataframe
        df_Q_Cum_day = np.array([pred_day, fittedParameters[0], fittedParameters[1], fittedParameters[2], prediction, deviation, abs_deviation])
        df_Q_Cum_day = pd.DataFrame(df_Q_Cum_day.reshape(-1, len(df_Q_Cum_day)),columns=column_names)
        df_Q_Cum = df_Q_Cum.append(df_Q_Cum_day)

    df_Q_Cum = df_Q_Cum.reset_index(drop = True)
    
    return df_Q_Cum

def Q_PPM(df):  
    """
    Use Q_PPM method to generate predictions and calculate deviations.
    Input: Individual patient data
    Output: Q_PPM results
    """
    # Create dataframe for Q-PPM
    column_names = ['prediction day', 'a', 'b', 'c', 'prediction', 'deviation', 'abs deviation']
    df_Q_PPM = pd.DataFrame(columns = column_names)

    # Calculate deviations and predictions for the first time:
    # Fill in prediction day, quadratic fit, prediction, deviation for first prediction
    day_num = 3 # Prediction starts at index 3, after taking 3 sets of unique data points
    pred_day = df['Day #'][day_num]

    # Find coefficients of quadratic fit
    fittedParameters = (np.polyfit(df["Eff 24h Tac Dose"][0:day_num], df["Tac level (prior to am dose)"][0:day_num], 2))

    # Calculate prediction based on quad fit
    prediction = np.polyval(fittedParameters, df["Eff 24h Tac Dose"][day_num])

    # Calculate deviation from prediction
    deviation = prediction - df["Tac level (prior to am dose)"][day_num]
    abs_deviation = abs(deviation)

    # Add details of first prediction into dataframe
    df_Q_PPM_day = np.array([pred_day, fittedParameters[0], fittedParameters[1], fittedParameters[2], prediction, deviation, abs_deviation])
    df_Q_PPM_day = pd.DataFrame(df_Q_PPM_day.reshape(-1, len(df_Q_PPM_day)),columns=column_names)
    df_Q_PPM = df_Q_PPM.append(df_Q_PPM_day)

    # Add subsequent predictions
    for day_num in range(4, len(df)):
        pred_day, a, b = df_Q_PPM["prediction day"].iloc[-1] + 1, df_Q_PPM['a'].iloc[-1], df_Q_PPM['b'].iloc[-1]
        c = df_Q_PPM['c'].iloc[-1] - df_Q_PPM['deviation'].iloc[-1]
        fittedParameters = a, b, c
        prediction = np.polyval(fittedParameters, df["Eff 24h Tac Dose"][day_num])
        deviation = prediction - df["Tac level (prior to am dose)"][day_num]
        abs_deviation = abs(deviation)

        df_Q_PPM_day = np.array([pred_day, a, b, c, prediction, deviation, abs_deviation])
        df_Q_PPM_day = pd.DataFrame(df_Q_PPM_day.reshape(-1, len(df_Q_PPM_day)),columns=column_names)
        df_Q_PPM = df_Q_PPM.append(df_Q_PPM_day)

    df_Q_PPM = df_Q_PPM.reset_index(drop = True)
    
    return df_Q_PPM

def Q_RW(df):
    """
    Use Q_RW method to generate predictions and calculate deviations.
    Input: Individual patient data
    Output: Q_RW results
    """
    # Create dataframe for Q-RW
    column_names = ['prediction day', 'a', 'b', 'c', 'prediction', 'deviation', 'abs deviation']
    df_Q_RW = pd.DataFrame(columns = column_names)

    for day_num in range(3, len(df)): 
        # Find prediction day
        pred_day = int(df["Day #"][day_num])

        # Find coefficients of quadratic fit
        fittedParameters = (np.polyfit(df["Eff 24h Tac Dose"][day_num-3:day_num], df["Tac level (prior to am dose)"][day_num-3:day_num], 2))

        # Calculate prediction based on quad fit
        prediction = np.polyval(fittedParameters, df["Eff 24h Tac Dose"][day_num])

        # Calculate deviation from prediction
        deviation = prediction - df["Tac level (prior to am dose)"][day_num]
        abs_deviation = abs(deviation)

        # Add prediction into dataframe
        df_Q_RW_day = np.array([pred_day, fittedParameters[0], fittedParameters[1], fittedParameters[2], prediction, deviation, abs_deviation])
        df_Q_RW_day = pd.DataFrame(df_Q_RW_day.reshape(-1, len(df_Q_RW_day)),columns=column_names)
        df_Q_RW = df_Q_RW.append(df_Q_RW_day)

    df_Q_RW = df_Q_RW.reset_index(drop = True)
    
    return df_Q_RW

def L_Cum(df):
    """
    Use L_Cum method to generate predictions and calculate deviations.
    Input: Individual patient data
    Output: L_Cum results
    """
    # Create dataframe for L-Cum
    column_names = ['prediction day', 'a', 'b', 'prediction', 'deviation', 'abs deviation']
    df_L_Cum = pd.DataFrame(columns = column_names)

    for day_num in range(2, len(df)):
        pred_day = day_num + 2 

        # Find coefficients of quadratic fit
        fittedParameters = (np.polyfit(df["Eff 24h Tac Dose"][0:day_num], df["Tac level (prior to am dose)"][0:day_num], 1))

        # Calculate prediction based on quad fit
        prediction = np.polyval(fittedParameters, df["Eff 24h Tac Dose"][day_num])

        # Calculate deviation from prediction
        deviation = prediction - df["Tac level (prior to am dose)"][day_num]
        abs_deviation = abs(deviation)

        # Add the prediction day, coefficients, prediction, and deviation below dataframe
        df_L_Cum_day = np.array([pred_day, fittedParameters[0], fittedParameters[1], prediction, deviation, abs_deviation])
        df_L_Cum_day = pd.DataFrame(df_L_Cum_day.reshape(-1, len(df_L_Cum_day)),columns=column_names)
        df_L_Cum = df_L_Cum.append(df_L_Cum_day)

    df_L_Cum = df_L_Cum.reset_index(drop = True)

    return df_L_Cum

def L_PPM(df):
    """
    Use L_PPM method to generate predictions and calculate deviations.
    Input: Individual patient data
    Output: L_PPM results
    """
    # Create dataframe for L-PPM
    column_names = ['prediction day', 'a', 'b', 'prediction', 'deviation', 'abs deviation']
    df_L_PPM = pd.DataFrame(columns = column_names)

    # Fill in prediction day, quadratic fit, prediction, deviation for prediction of day 5
    pred_day = 4
    day_num = 2

    # Find coefficients of quadratic fit
    fittedParameters = (np.polyfit(df["Eff 24h Tac Dose"][0:day_num], df["Tac level (prior to am dose)"][0:day_num], 1))

    # Calculate prediction based on quad fit
    prediction = np.polyval(fittedParameters, df["Eff 24h Tac Dose"][day_num])

    # Calculate deviation from prediction
    deviation = prediction - df["Tac level (prior to am dose)"][day_num]
    abs_deviation = abs(deviation)

    # Add prediction of day 5 into dataframe
    df_L_PPM_day = np.array([pred_day, fittedParameters[0], fittedParameters[1], prediction, deviation, abs_deviation])
    df_L_PPM_day = pd.DataFrame(df_L_PPM_day.reshape(-1, len(df_L_PPM_day)),columns=column_names)
    df_L_PPM = df_L_PPM.append(df_L_PPM_day)

    # Add subsequent predictions
    for day_num in range(3, len(df)):
        pred_day, a = df_L_PPM["prediction day"].iloc[-1] + 1, df_L_PPM['a'].iloc[-1]
        b = df_L_PPM['b'].iloc[-1] - df_L_PPM['deviation'].iloc[-1]
        fittedParameters = a, b
        prediction = np.polyval(fittedParameters, df["Eff 24h Tac Dose"][day_num])
        deviation = prediction - df["Tac level (prior to am dose)"][day_num]
        abs_deviation = abs(deviation)

        df_L_PPM_day = np.array([pred_day, a, b, prediction, deviation, abs_deviation])
        df_L_PPM_day = pd.DataFrame(df_L_PPM_day.reshape(-1, len(df_L_PPM_day)),columns=column_names)
        df_L_PPM = df_L_PPM.append(df_L_PPM_day)

    df_L_PPM = df_L_PPM.reset_index(drop = True)
    return df_L_PPM

def L_RW(df):
    """
    Use L_RW method to generate predictions and calculate deviations.
    Input: Individual patient data
    Output: L_RW results
    """
    # Create dataframe for L-RW
    column_names = ['prediction day', 'a', 'b', 'prediction', 'deviation', 'abs deviation']
    df_L_RW = pd.DataFrame(columns = column_names)

    for day_num in range(2, len(df)): 
        # Find prediction day
        pred_day = int(df["Day #"][day_num])

        # Find coefficients of quadratic fit
        fittedParameters = (np.polyfit(df["Eff 24h Tac Dose"][day_num-2:day_num], df["Tac level (prior to am dose)"][day_num-2:day_num], 1))

        # Calculate prediction based on quad fit
        prediction = np.polyval(fittedParameters, df["Eff 24h Tac Dose"][day_num])

        # Calculate deviation from prediction
        deviation = prediction - df["Tac level (prior to am dose)"][day_num]
        abs_deviation = abs(deviation)

        # Add prediction into dataframe
        df_L_RW_day = np.array([pred_day, fittedParameters[0], fittedParameters[1], prediction, deviation, abs_deviation])
        df_L_RW_day = pd.DataFrame(df_L_RW_day.reshape(-1, len(df_L_RW_day)),columns=column_names)
        df_L_RW = df_L_RW.append(df_L_RW_day)

    df_L_RW = df_L_RW.reset_index(drop = True)
    return df_L_RW


# Plotting

def deviation_without_intercept(df_Q_Cum, df_Q_PPM, df_Q_RW,
                                df_L_Cum, df_L_PPM, df_L_RW):
    """ Plot deviation for all methods without intercept """
    # Plot x = prediction day, y = deviation, by method (color)

    # for col in range(df_deviation.shape[1]):
    #     plt.scatter(df_deviation['pred_day'], df_deviation.iloc[:, col])

    df_deviation.plot(x='pred_day', 
                      y=['Q_Cum', 'Q_PPM', 'Q_RW', 'L_Cum', 'L_PPM', 'L_RW'], 
                      kind="line")
    plt.xlabel("Day of Prediction")
    plt.ylabel("Deviation")
    plt.title("Deviation of Prediction from Actual Value")
    plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")

    # plt.savefig("test.png", format="png", dpi=300)
    plt.tight_layout()
    # plt.savefig('myfile.png', bbox_inches="tight", dpi=300)
    plt.show()

def plot_mean_dev_without_intercept(df_deviation):
    # Plot mean deviation

    methods = ['Q_Cum', 'Q_PPM', 'Q_RW', 'L_Cum', 'L_PPM', 'L_RW']
    x_pos = np.arange(len(methods))
    CTEs = [np.mean(df_deviation['Q_Cum']), 
            np.mean(df_deviation['Q_PPM']),
           np.mean(df_deviation['Q_RW']),
           np.mean(df_deviation['L_Cum']),
           np.mean(df_deviation['L_PPM']),
           np.mean(df_deviation['L_RW'])]
    error = [np.std(df_deviation['Q_Cum']), 
            np.std(df_deviation['Q_PPM']),
           np.std(df_deviation['Q_RW']),
           np.std(df_deviation['L_Cum']),
           np.std(df_deviation['L_PPM']),
           np.std(df_deviation['L_RW'])]

    # Build the plot
    fig, ax = plt.subplots()
    ax.bar(x_pos, CTEs, yerr=error, align='center', alpha=0.5, ecolor='black', capsize=10)
    ax.set_ylabel('Deviation (Mean \u00B1 SD)')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.set_title('Deviation of Predicted from Actual Value (Mean \u00B1 SD)')
    ax.yaxis.grid(True)

    # Save the figure and show
    plt.tight_layout()
    # plt.savefig('Mean Deviation.png', bbox_inches="tight", dpi=300)
    plt.show()

def median_dev_without_intercept(df_deviation):
    # Plot median of deviation
    methods = ['Q_Cum', 'Q_PPM', 'Q_RW', 'L_Cum', 'L_PPM', 'L_RW']
    x_pos = np.arange(len(methods))

    data = [df_deviation['Q_Cum'],
           df_deviation['Q_PPM'],
           df_deviation['Q_RW'],
           df_deviation['L_Cum'],
           df_deviation['L_PPM'],
           df_deviation['L_RW']]

    fig, ax = plt.subplots()
    ax.set_title('Deviation of Predicted from Actual Value (Median)')
    ax.boxplot(data)
    ax.set_xticklabels(methods)
    # plt.ylabel('Deviation (Median)')
    plt.savefig('Median Deviation.png', bbox_inches="tight", dpi=300)
    plt.show()