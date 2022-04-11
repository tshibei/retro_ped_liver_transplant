import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import numpy as np
from scipy import stats
import seaborn as sns
from functools import reduce
pd.options.mode.chained_assignment = None 
from statistics import mean
from scipy.optimize import curve_fit

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
    df['Eff 24h Tac Dose'] = df['Eff 24h Tac Dose'].astype(str).str.replace('ng', '')
    df['Eff 24h Tac Dose'] = df['Eff 24h Tac Dose'].astype(float)

    return df

# Data selection

def keep_ideal_data(df):
    """
    Remove rows with non-ideal data, including missing tac level and/or tac dose, 
    <2 tac level, multiple blood draws. Then keep the longest consecutive chunk
    based on number of days. 

    Input: Dataframe of individual patient
    Output: Dataframe with the longest chunk of consecutive ideal data.
    """
    # Create copy of df to manipulate and find largest consec non-NA chunk
    df_temp = df.copy()

    # Create boolean column of data to remove
    # Including NA, <2 tac level, multiple blood draws
    df_temp['non_ideal'] = (df_temp.isnull().values.any(axis=1))  | \
                               (df_temp['Tac level (prior to am dose)'] == '<2') | \
                               (df_temp["Tac level (prior to am dose)"].astype(str).str.contains("/"))

    # Create index column
    df_temp.reset_index(inplace=True) 

    # Find cumulative sum of data to be removed for each index row
    df_cum_sum_non_ideal = df_temp['non_ideal'].cumsum()     

    # Find number of consecutive non-NA
    df_temp = df_temp.groupby(df_cum_sum_non_ideal).agg({'index': ['count', 'min', 'max']})

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

# Combine calibration and prediction data into one dataframe

def cal_pred_data(df, patient, cal_pred, patients_to_exclude, deg):
    """
    Find calibration points and combine calibration data with the rest of data,
    to perform CURATE methods later.
    
    Input: 
    df - dataframe
    patient - string of patient number
    deg - degree of fitting polynomial (1 for linear, 2 for quadratic)

    Output: 
    Dataframe of calibration data and data for subsequent predictions.
    Print patients with insufficient data for calibration and <3 predictions

    """
    # Create index column
    df[patient] = df[patient].reset_index(drop=False) 

    # Create boolean column to check if tac dose diff from next row
    df[patient]['diff_from_next'] = \
            (df[patient]['Eff 24h Tac Dose'] != df[patient]['Eff 24h Tac Dose'].shift(-1))

    # Find indexes of last rows of first 2 unique doses
    last_unique_doses_idx = [i for i, x in enumerate(df[patient].diff_from_next) if x]

    # Create boolean column to check if tac dose diff from previous row
    df[patient]['diff_from_prev'] = \
            (df[patient]['Eff 24h Tac Dose'] != df[patient]['Eff 24h Tac Dose'].shift(1))

    # Find indexes of first rows of third unique dose
    first_unique_doses_idx = [i for i, x in enumerate(df[patient].diff_from_prev) if x]

    # The boolean checks created useless index, diff_from_next and diff_from_prev columns,
    # drop them
    df[patient] = df[patient].drop(['index', 'diff_from_next', 'diff_from_prev'], axis=1)

    # Combine calibration and prediction rows
    cal_pred[patient] = []
    
    # Do for quadratic method
    if deg == 2:
        if df[patient]['Eff 24h Tac Dose'].nunique() > 2:
            # If third cal point is the same as first 2 cal points, keep looking
            first_cal_dose = df[patient]['Eff 24h Tac Dose'][last_unique_doses_idx[0]]
            second_cal_dose = df[patient]['Eff 24h Tac Dose'][last_unique_doses_idx[1]]
            n = 2
            for i in range(n, len(df[patient])+1):
                third_cal_dose = df[patient]['Eff 24h Tac Dose'][first_unique_doses_idx[n]]
                if (third_cal_dose == first_cal_dose) | (third_cal_dose == second_cal_dose):
                    n = n + 1

            first_cal_point = pd.DataFrame(df[patient].iloc[last_unique_doses_idx[0],:]).T
            second_cal_point = pd.DataFrame(df[patient].iloc[last_unique_doses_idx[1],:]).T
            third_cal_point = pd.DataFrame(df[patient].iloc[first_unique_doses_idx[n],:]).T
            rest_of_data = df[patient].iloc[first_unique_doses_idx[n]+1:,:]
            cal_pred[patient] = pd.concat([first_cal_point, second_cal_point, third_cal_point, 
                                        rest_of_data]).reset_index(drop=True)
        else:
            patients_to_exclude.append(str(patient))
            print(patient, ": Insufficient unique dose-response pairs for quadratic calibration!")

    # Do for linear method
    if deg == 1:
        if df[patient]['Eff 24h Tac Dose'].nunique() > 1:
            first_cal_point = pd.DataFrame(df[patient].iloc[last_unique_doses_idx[0],:]).T
            second_cal_point = pd.DataFrame(df[patient].iloc[first_unique_doses_idx[1],:]).T
            rest_of_data = df[patient].iloc[first_unique_doses_idx[1]+1:,:]
            cal_pred[patient] = pd.concat([first_cal_point, second_cal_point, 
                                        rest_of_data]).reset_index(drop=True)
        else:
            patients_to_exclude.append(str(patient))
            print(patient, ": Insufficient unique dose-response pairs for linear calibration!")

    # Print error msg if number of predictions is less than 3
    if df[patient]['Eff 24h Tac Dose'].nunique() < 3:
        pass # there are insufficient data for calibration already, don't need
             # this error msg
    elif len(cal_pred[patient]) - (deg + 1) < 3:
        patients_to_exclude.append(str(patient))
        if deg == 1:
            error_string = '(for linear)'
        else:
            error_string = '(for quadratic)'
        print(patient, ": No. of predictions ", error_string," is <3: ", len(cal_pred[patient]) - (deg + 1))

    return cal_pred[patient]

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
    # 1. Create input dataframe for prediction
    
    # Create column names
    column_names = ['Pred_Day'] + ['Dose_' + str(i) for i in range(1, len(df) + 1)] + \
                   ['Response_' + str(i) for i in range(1, len(df) + 1)] + \
                   ['New_Dose', 'New_Response']

    # Create dataframe
    df_Q_Cum_input = pd.DataFrame(columns = column_names)

    # Loop through rows from first to last prediction
    for i in range(3, len(df)):

        # Find doses and responses from previous rows
        doses = df["Eff 24h Tac Dose"][0:i].to_numpy()
        responses = df["Tac level (prior to am dose)"][0:i].to_numpy()

        # Create temporary dataframe of doses, responses, new dose, new response
        column_names = ['Pred_Day'] + ['Dose_' + str(i) for i in range(1, len(df) + 1)] + \
                    ['Response_' + str(i) for i in range(1, len(df) + 1)] + \
                    ['New_Dose', 'New_Response']

        df_temp = pd.DataFrame(columns = column_names)

        # Fill in values in one row in df_temp
        df_temp.loc[0, 'Pred_Day'] = df["Day #"][i]
        df_temp.loc[0, 'Dose_1':'Dose_' + str(i)] = doses
        df_temp.loc[0, 'Response_1':'Response_' + str(i)] = responses
        df_temp.loc[0, 'New_Dose'] = df["Eff 24h Tac Dose"][i]
        df_temp.loc[0, 'New_Response'] = df["Tac level (prior to am dose)"][i]

        # Concat input dataframe with df_temp
        df_Q_Cum_input = pd.concat([df_Q_Cum_input, df_temp])
        df_Q_Cum_input = df_Q_Cum_input.reset_index(drop=True)

    # 2. Create dataframe for Q-Cum results
    column_names = ['prediction day', 'a', 'b', 'c', 'prediction', 'deviation', 'abs deviation']
    df_Q_Cum = pd.DataFrame(columns = column_names)

    for i in range(0, len(df_Q_Cum_input)):
        pred_day = int(df_Q_Cum_input["Pred_Day"][i])

        # Find coefficients of quadratic fit
        fittedParameters = (np.polyfit(df_Q_Cum_input.loc[i, 'Dose_1':'Dose_' + str(i+3)].astype(float), df_Q_Cum_input.loc[i, 'Dose_1':'Dose_' + str(i+3)].astype(float), 2))

        # Calculate prediction based on quad fit
        prediction = np.polyval(fittedParameters, df_Q_Cum_input["New_Dose"][i])

        # Calculate deviation from prediction
        deviation = prediction - df_Q_Cum_input["New_Response"][i]
        abs_deviation = abs(deviation)

        # Add the prediction day, coefficients, prediction, and deviation below dataframe
        df_Q_Cum_day = np.array([pred_day, fittedParameters[0], fittedParameters[1], fittedParameters[2], prediction, deviation, abs_deviation])
        df_Q_Cum_day = pd.DataFrame(df_Q_Cum_day.reshape(-1, len(df_Q_Cum_day)),columns=column_names)
        df_Q_Cum = df_Q_Cum.append(df_Q_Cum_day)

    df_Q_Cum = df_Q_Cum.reset_index(drop = True)
    
    return df_Q_Cum_input, df_Q_Cum

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
    fittedParameters = (np.polyfit(df["Eff 24h Tac Dose"][0:day_num].astype(float), df["Tac level (prior to am dose)"][0:day_num].astype(float), 2))

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

def select_RW_data(cal_pred_dataframe, num_of_data_pairs):
    """
    For each prediction day, find the previous 2/3 data pairs (for linear/quadratic method) for rolling window prediction.
    
    Input: Calibration and prediction data of a patient (e.g. cal_pred['114'])
    Output: Dose-response pairs for rolling window prediction
    """

    values = []
    indices = []
    fittedParameters = []
    df_input = pd.DataFrame(columns = ['Pred_Day', 'Dose_1', 'Dose_2', 'Dose_3',
                                      'Response_1', 'Response_2', 'Response_3', 
                                       'New_Dose', 'New_Response'])

    # Find values and indices of dose-response pairs for RW:

    # Loop through rows
    for i in range(0, len(cal_pred_dataframe) - 1):

        # Define new dose and prediction day
        new_dose = cal_pred_dataframe['Eff 24h Tac Dose'][i]
        day = cal_pred_dataframe['Day #'][i+1]

        # Add values and indices of first three unique doses into list
        if len(values) < num_of_data_pairs:

            # Check if new dose in values list
            # If not, append new dose into value list and append index into indices list
            if new_dose not in values:
                values.append(new_dose)
                indices.append(i)

        else:
            # Check if new dose in value list
            # If yes: remove value and index of repeat, and append new dose and index to list.
            if new_dose in values:
                repeated_dose_index = values.index(new_dose)
                values.pop(repeated_dose_index)
                indices.pop(repeated_dose_index)

                values.append(new_dose)
                indices.append(i)

            # If no: remove the first element of list and append value and index of new dose
            else:
                values.pop(0)
                indices.pop(0)

                values.append(new_dose)
                indices.append(i)

        # Prepare dataframe of dose-response pairs for prediction
        if len(indices) == num_of_data_pairs:

            if num_of_data_pairs == 3:

                dict = {'Pred_Day': day,
                        'Dose_1': cal_pred_dataframe['Eff 24h Tac Dose'][indices[0]],
                        'Dose_2': cal_pred_dataframe['Eff 24h Tac Dose'][indices[1]],
                        'Dose_3': cal_pred_dataframe['Eff 24h Tac Dose'][indices[2]],
                        'Response_1': cal_pred_dataframe['Tac level (prior to am dose)'][indices[0]],
                        'Response_2': cal_pred_dataframe['Tac level (prior to am dose)'][indices[1]],
                        'Response_3': cal_pred_dataframe['Tac level (prior to am dose)'][indices[2]],
                        'New_Dose': cal_pred_dataframe['Eff 24h Tac Dose'][i+1],
                        'New_Response': cal_pred_dataframe['Tac level (prior to am dose)'][i+1]}

                df_input = df_input.append(dict, ignore_index=True)
            
            else:

                dict = {'Pred_Day': day,
                        'Dose_1': cal_pred_dataframe['Eff 24h Tac Dose'][indices[0]],
                        'Dose_2': cal_pred_dataframe['Eff 24h Tac Dose'][indices[1]],
                        'Response_1': cal_pred_dataframe['Tac level (prior to am dose)'][indices[0]],
                        'Response_2': cal_pred_dataframe['Tac level (prior to am dose)'][indices[1]],
                        'New_Dose': cal_pred_dataframe['Eff 24h Tac Dose'][i+1],
                        'New_Response': cal_pred_dataframe['Tac level (prior to am dose)'][i+1]}

                df_input = df_input.append(dict, ignore_index=True)
        
    return df_input

def RW(df_input, patient, df_RW, num_of_data_pairs):
    """
    Use Rolling Window method to generate predictions and calculate deviations.
    
    Input: Dose response pairs for prediction of individual patient
    Output: Rolling Window results
    """
    # Perform RW method:

    # Define dataframe output for RW method
    df_RW[patient] = pd.DataFrame(columns = ['prediction day', 'a', 'b', 'c', 'prediction', 'deviation', 'abs deviation'])

    # Loop through rows
    for i in range(0, len(df_input)):

        # Perform quadratic fit
        if num_of_data_pairs == 3:
            x = df_input.loc[i, ['Dose_1', 'Dose_2', 'Dose_3']]
            y = df_input.loc[i, ['Response_1', 'Response_2', 'Response_3']]
        else: 
            x = df_input.loc[i, ['Dose_1', 'Dose_2']]
            y = df_input.loc[i, ['Response_1', 'Response_2']]
        fittedParameters = (np.polyfit(x, y, num_of_data_pairs - 1))

        # Predict new response
        prediction = np.polyval(fittedParameters, df_input.loc[i, 'New_Dose'])

        # Calculate deviation
        deviation = prediction - df_input.loc[i, 'New_Response']
        abs_deviation = abs(deviation)

        # Append results into dataframe
        if num_of_data_pairs == 3:
            dict = {'prediction day': df_input.loc[i, 'Pred_Day'],
                'a': fittedParameters[0],
                'b': fittedParameters[1],
                'c': fittedParameters[2],
                'prediction': prediction,
                'deviation': deviation,
                'abs deviation': abs_deviation}
        else:
            dict = {'prediction day': df_input.loc[i, 'Pred_Day'],
                'a': fittedParameters[0],
                'b': fittedParameters[1],
                'prediction': prediction,
                'deviation': deviation,
                'abs deviation': abs_deviation}

        df_RW[patient] = df_RW[patient].append(dict, ignore_index=True)
        
    return df_RW[patient]

def L_Cum(df):
    """
    Use L_Cum method to generate predictions and calculate deviations.
    Input: Individual patient data
    Output: L_Cum results
    """
    # 1. Create input dataframe for prediction
    
    # Create column names
    column_names = ['Pred_Day'] + ['Dose_' + str(i) for i in range(1, len(df) + 1)] + \
                   ['Response_' + str(i) for i in range(1, len(df) + 1)] + \
                   ['New_Dose', 'New_Response']

    # Create dataframe
    df_L_Cum_input = pd.DataFrame(columns = column_names)

    # Loop through rows from first to last prediction
    for i in range(2, len(df)):

        # Find doses and responses from previous rows
        doses = df["Eff 24h Tac Dose"][0:i].to_numpy()
        responses = df["Tac level (prior to am dose)"][0:i].to_numpy()

        # Create temporary dataframe of doses, responses, new dose, new response
        column_names = ['Pred_Day'] + ['Dose_' + str(i) for i in range(1, len(df) + 1)] + \
                    ['Response_' + str(i) for i in range(1, len(df) + 1)] + \
                    ['New_Dose', 'New_Response']

        df_temp = pd.DataFrame(columns = column_names)

        # Fill in values in one row in df_temp
        df_temp.loc[0, 'Pred_Day'] = df["Day #"][i]
        df_temp.loc[0, 'Dose_1':'Dose_' + str(i)] = doses
        df_temp.loc[0, 'Response_1':'Response_' + str(i)] = responses
        df_temp.loc[0, 'New_Dose'] = df["Eff 24h Tac Dose"][i]
        df_temp.loc[0, 'New_Response'] = df["Tac level (prior to am dose)"][i]

        # Concat input dataframe with df_temp
        df_L_Cum_input = pd.concat([df_L_Cum_input, df_temp])
        df_L_Cum_input = df_L_Cum_input.reset_index(drop=True)
    
    # 2. Create dataframe for L-Cum results 
    column_names = ['prediction day', 'a', 'b', 'prediction', 'deviation', 'abs deviation']
    df_L_Cum = pd.DataFrame(columns = column_names)

    for i in range(0, len(df_L_Cum_input)):
        pred_day = int(df_L_Cum_input["Pred_Day"][i])

        # Find coefficients of quadratic fit
        fittedParameters = (np.polyfit(df_L_Cum_input.loc[i, 'Dose_1':'Dose_' + str(i+2)].astype(float), df_L_Cum_input.loc[i, 'Dose_1':'Dose_' + str(i+2)].astype(float), 1))

        # Calculate prediction based on quad fit
        prediction = np.polyval(fittedParameters, df_L_Cum_input["New_Dose"][i])

        # Calculate deviation from prediction
        deviation = prediction - df_L_Cum_input["New_Response"][i]
        abs_deviation = abs(deviation)

        # Add the prediction day, coefficients, prediction, and deviation below dataframe
        df_L_Cum_day = np.array([pred_day, fittedParameters[0], fittedParameters[1], prediction, deviation, abs_deviation])
        df_L_Cum_day = pd.DataFrame(df_L_Cum_day.reshape(-1, len(df_L_Cum_day)),columns=column_names)
        df_L_Cum = df_L_Cum.append(df_L_Cum_day)

    df_L_Cum = df_L_Cum.reset_index(drop = True)

    return df_L_Cum_input, df_L_Cum

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
    fittedParameters = (np.polyfit(df["Eff 24h Tac Dose"][0:day_num].astype(float), df["Tac level (prior to am dose)"][0:day_num].astype(float), 1))

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
        fittedParameters = (np.polyfit(df["Eff 24h Tac Dose"][day_num-2:day_num].astype(float), df["Tac level (prior to am dose)"][day_num-2:day_num].astype(float), 1))

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

def Q_Cum_origin_int(df):
    """
    Use Q_Cum_origin_int method to generate predictions and calculate deviations.
    Input: Individual patient data
    Output: Q_Cum_origin_int results
    """
    # 1. Create input dataframe for prediction
    
    # Create column names
    column_names = ['Pred_Day'] + ['Dose_' + str(i) for i in range(1, len(df) + 1)] + \
                   ['Response_' + str(i) for i in range(1, len(df) + 1)] + \
                   ['New_Dose', 'New_Response']

    # Create dataframe
    df_Q_Cum_origin_int_input = pd.DataFrame(columns = column_names)

    # Loop through rows from first to last prediction
    for i in range(3, len(df)):

        # Find doses and responses from previous rows
        doses = df["Eff 24h Tac Dose"][0:i].to_numpy()
        responses = df["Tac level (prior to am dose)"][0:i].to_numpy()

        # Create temporary dataframe of doses, responses, new dose, new response
        column_names = ['Pred_Day'] + ['Dose_' + str(i) for i in range(1, len(df) + 1)] + \
                    ['Response_' + str(i) for i in range(1, len(df) + 1)] + \
                    ['New_Dose', 'New_Response']

        df_temp = pd.DataFrame(columns = column_names)

        # Fill in values in one row in df_temp
        df_temp.loc[0, 'Pred_Day'] = df["Day #"][i]
        df_temp.loc[0, 'Dose_1':'Dose_' + str(i)] = doses
        df_temp.loc[0, 'Response_1':'Response_' + str(i)] = responses
        df_temp.loc[0, 'New_Dose'] = df["Eff 24h Tac Dose"][i]
        df_temp.loc[0, 'New_Response'] = df["Tac level (prior to am dose)"][i]

        # Concat input dataframe with df_temp
        df_Q_Cum_origin_int_input = pd.concat([df_Q_Cum_origin_int_input, df_temp])
        df_Q_Cum_origin_int_input = df_Q_Cum_origin_int_input.reset_index(drop=True)

    def f(x, a, b):
        return a*x**2 + b*x

    # 2. Create dataframe for Q-Cum results
    column_names = ['prediction day', 'a', 'b', 'c', 'prediction', 'deviation', 'abs deviation']
    df_Q_Cum_origin_int = pd.DataFrame(columns = column_names)

    for day_num in range(3, len(df)):
        pred_day = int(df["Day #"][day_num])

        # Find coefficients of quadratic fit
        # x = df["Eff 24h Tac Dose"][0:day_num].to_numpy()
        # y = df["Tac level (prior to am dose)"][0:day_num].to_numpy()

        x = df_Q_Cum_origin_int_input.loc[day_num-3, 'Dose_1':'Dose_' + str(day_num)].to_numpy()
        y = df_Q_Cum_origin_int_input.loc[day_num-3, 'Response_1':'Response_' + str(day_num)].to_numpy()

        popt, pcov = curve_fit(f, x, y)

        fittedParameters = [popt[0], popt[1], 0]

        # Calculate prediction based on quad fit
        prediction = np.polyval(fittedParameters, df["Eff 24h Tac Dose"][day_num])

        # Calculate deviation from prediction
        deviation = prediction - df["Tac level (prior to am dose)"][day_num]
        abs_deviation = abs(deviation)

        # Add the prediction day, coefficients, prediction, and deviation below dataframe
        df_Q_Cum_origin_int_day = np.array([pred_day, fittedParameters[0], fittedParameters[1], fittedParameters[2], prediction, deviation, abs_deviation])
        df_Q_Cum_origin_int_day = pd.DataFrame(df_Q_Cum_origin_int_day.reshape(-1, len(df_Q_Cum_origin_int_day)),columns=column_names)
        df_Q_Cum_origin_int = df_Q_Cum_origin_int.append(df_Q_Cum_origin_int_day)

    df_Q_Cum_origin_int = df_Q_Cum_origin_int.reset_index(drop = True)

    return df_Q_Cum_origin_int_input, df_Q_Cum_origin_int

def Q_PPM_origin_int(df):
    """
    Use Q_PPM_origin_int method to generate predictions and calculate deviations.
    Input: Individual patient data
    Output: Q_PPM_origin_int results
    """
    from scipy.optimize import curve_fit
    # Create dataframe for Q-PPM_origin_int
    column_names = ['prediction day', 'a', 'b', 'c', 'prediction', 'deviation', 'abs deviation']
    df_Q_PPM_origin_int = pd.DataFrame(columns = column_names)

    # Fill in prediction day, quadratic fit, prediction, deviation for prediction of day 5
    pred_day = 5
    day_num = 3

    def f(x, a, b):
        return a*x**2 + b*x

    # Find coefficients of quadratic fit
    x = df["Eff 24h Tac Dose"][0:day_num].to_numpy()
    y = df["Tac level (prior to am dose)"][0:day_num].to_numpy()
    popt, pcov = curve_fit(f, x, y)

    fittedParameters = [popt[0], popt[1], 0]

    # Calculate prediction based on quad fit
    prediction = np.polyval(fittedParameters, df["Eff 24h Tac Dose"][day_num])

    # Calculate deviation from prediction
    deviation = prediction - df["Tac level (prior to am dose)"][day_num]
    abs_deviation = abs(deviation)

    # Add prediction of day 5 into dataframe
    df_Q_PPM_origin_int_day = np.array([pred_day, fittedParameters[0], fittedParameters[1], fittedParameters[2], prediction, deviation, abs_deviation])
    df_Q_PPM_origin_int_day = pd.DataFrame(df_Q_PPM_origin_int_day.reshape(-1, len(df_Q_PPM_origin_int_day)),columns=column_names)
    df_Q_PPM_origin_int = df_Q_PPM_origin_int.append(df_Q_PPM_origin_int_day)

    # Add subsequent predictions
    for day_num in range(4, len(df)):
        pred_day, a, b = df_Q_PPM_origin_int["prediction day"].iloc[-1] + 1, df_Q_PPM_origin_int['a'].iloc[-1], df_Q_PPM_origin_int['b'].iloc[-1]
        c = df_Q_PPM_origin_int['c'].iloc[-1] - df_Q_PPM_origin_int['deviation'].iloc[-1]
        fittedParameters = a, b, c
        prediction = np.polyval(fittedParameters, df["Eff 24h Tac Dose"][day_num])
        deviation = prediction - df["Tac level (prior to am dose)"][day_num]
        abs_deviation = abs(deviation)

        df_Q_PPM_origin_int_day = np.array([pred_day, a, b, c, prediction, deviation, abs_deviation])
        df_Q_PPM_origin_int_day = pd.DataFrame(df_Q_PPM_origin_int_day.reshape(-1, len(df_Q_PPM_origin_int_day)),columns=column_names)
        df_Q_PPM_origin_int = df_Q_PPM_origin_int.append(df_Q_PPM_origin_int_day)

    df_Q_PPM_origin_int = df_Q_PPM_origin_int.reset_index(drop = True)
    return df_Q_PPM_origin_int

def Q_RW_origin_int(df):
    """
    Use Q_RW_origin_int method to generate predictions and calculate deviations.
    Input: Individual patient data
    Output: Q_RW_origin_int results
    """
    from scipy.optimize import curve_fit
    
    # Create dataframe for Q-RW
    column_names = ['prediction day', 'a', 'b', 'c', 'prediction', 'deviation', 'abs deviation']
    df_Q_RW_origin_int = pd.DataFrame(columns = column_names)

    for day_num in range(3, len(df)): 
        # Find prediction day
        pred_day = int(df["Day #"][day_num])

        # Find coefficients of quadratic fit
        def f(x, a, b):
            return a*x**2 + b*x

        x = df["Eff 24h Tac Dose"][day_num-3:day_num].to_numpy()
        y = df["Tac level (prior to am dose)"][day_num-3:day_num].to_numpy()
        popt, pcov = curve_fit(f, x, y)

        fittedParameters = [popt[0], popt[1], 0]

        # Calculate prediction based on quad fit
        prediction = np.polyval(fittedParameters, df["Eff 24h Tac Dose"][day_num])

        # Calculate deviation from prediction
        deviation = prediction - df["Tac level (prior to am dose)"][day_num]
        abs_deviation = abs(deviation)

        # Add prediction into dataframe
        df_Q_RW_origin_int_day = np.array([pred_day, fittedParameters[0], fittedParameters[1], fittedParameters[2], prediction, deviation, abs_deviation])
        df_Q_RW_origin_int_day = pd.DataFrame(df_Q_RW_origin_int_day.reshape(-1, len(df_Q_RW_origin_int_day)),columns=column_names)
        df_Q_RW_origin_int = df_Q_RW_origin_int.append(df_Q_RW_origin_int_day)

    df_Q_RW_origin_int = df_Q_RW_origin_int.reset_index(drop = True)
    return df_Q_RW_origin_int

def L_Cum_origin_int(df):
    """
    Use L_Cum_origin_int method to generate predictions and calculate deviations.
    Input: Individual patient data
    Output: L_Cum_origin_int results
    """
    from scipy.optimize import curve_fit

    def f(x, a):
        return a*x

    # Create dataframe for L-Cum_origin_int
    column_names = ['prediction day', 'a', 'b', 'prediction', 'deviation', 'abs deviation']
    df_L_Cum_origin_int = pd.DataFrame(columns = column_names)

    for day_num in range(2, len(df)):
        pred_day = int(df["Day #"][day_num])

        # Find coefficients of quadratic fit
        x = df["Eff 24h Tac Dose"][0:day_num].to_numpy()
        y = df["Tac level (prior to am dose)"][0:day_num].to_numpy()
        popt, pcov = curve_fit(f, x, y)

        fittedParameters = [popt[0], 0]

        # Calculate prediction based on quad fit
        prediction = np.polyval(fittedParameters, df["Eff 24h Tac Dose"][day_num])

        # Calculate deviation from prediction
        deviation = prediction - df["Tac level (prior to am dose)"][day_num]
        abs_deviation = abs(deviation)

        # Add the prediction day, coefficients, prediction, and deviation below dataframe
        df_L_Cum_origin_int_day = np.array([pred_day, fittedParameters[0], fittedParameters[1], prediction, deviation, abs_deviation])
        df_L_Cum_origin_int_day = pd.DataFrame(df_L_Cum_origin_int_day.reshape(-1, len(df_L_Cum_origin_int_day)),columns=column_names)
        df_L_Cum_origin_int = df_L_Cum_origin_int.append(df_L_Cum_origin_int_day)

    df_L_Cum_origin_int = df_L_Cum_origin_int.reset_index(drop = True)
    return df_L_Cum_origin_int

def L_PPM_origin_int(df):
    """
    Use L_PPM_origin_int method to generate predictions and calculate deviations.
    Input: Individual patient data
    Output: L_PPM_origin_int results
    """
    from scipy.optimize import curve_fit
    
    # Create dataframe for L-PPM_origin_int
    column_names = ['prediction day', 'a', 'b', 'prediction', 'deviation', 'abs deviation']
    df_L_PPM_origin_int = pd.DataFrame(columns = column_names)

    # Fill in prediction day, linearratic fit, prediction, deviation for prediction of day 5
    pred_day = 4
    day_num = 2

    def f(x, a):
        return a*x
    
    # Find coefficients of linear fit
    x = df["Eff 24h Tac Dose"][0:day_num].to_numpy()
    y = df["Tac level (prior to am dose)"][0:day_num].to_numpy()
    popt, pcov = curve_fit(f, x, y)

    fittedParameters = [popt[0], 0]

    # Calculate prediction based on linear fit
    prediction = np.polyval(fittedParameters, df["Eff 24h Tac Dose"][day_num])

    # Calculate deviation from prediction
    deviation = prediction - df["Tac level (prior to am dose)"][day_num]
    abs_deviation = abs(deviation)

    # Add prediction of day 5 into dataframe
    df_L_PPM_origin_int_day = np.array([pred_day, fittedParameters[0], fittedParameters[1], prediction, deviation, abs_deviation])
    df_L_PPM_origin_int_day = pd.DataFrame(df_L_PPM_origin_int_day.reshape(-1, len(df_L_PPM_origin_int_day)),columns=column_names)
    df_L_PPM_origin_int = df_L_PPM_origin_int.append(df_L_PPM_origin_int_day)

    # Add subsequent predictions
    for day_num in range(3, len(df)):
        pred_day, a = df_L_PPM_origin_int["prediction day"].iloc[-1] + 1, df_L_PPM_origin_int['a'].iloc[-1]
        b = df_L_PPM_origin_int['b'].iloc[-1] - df_L_PPM_origin_int['deviation'].iloc[-1]
        fittedParameters = a, b
        prediction = np.polyval(fittedParameters, df["Eff 24h Tac Dose"][day_num])
        deviation = prediction - df["Tac level (prior to am dose)"][day_num]
        abs_deviation = abs(deviation)

        df_L_PPM_origin_int_day = np.array([pred_day, a, b, prediction, deviation, abs_deviation])
        df_L_PPM_origin_int_day = pd.DataFrame(df_L_PPM_origin_int_day.reshape(-1, len(df_L_PPM_origin_int_day)),columns=column_names)
        df_L_PPM_origin_int = df_L_PPM_origin_int.append(df_L_PPM_origin_int_day)

    df_L_PPM_origin_int = df_L_PPM_origin_int.reset_index(drop = True)
    return df_L_PPM_origin_int

def L_RW_origin_int(df):
    """
    Use L_RW_origin_int method to generate predictions and calculate deviations.
    Input: Individual patient data
    Output: L_RW_origin_int results
    """
    from scipy.optimize import curve_fit

    # Create dataframe for L-RW
    column_names = ['prediction day', 'a', 'b', 'prediction', 'deviation', 'abs deviation']
    df_L_RW_origin_int = pd.DataFrame(columns = column_names)

    for day_num in range(2, len(df)): 
        # Find prediction day
        pred_day = int(df["Day #"][day_num])

        # Find coefficients of linear fit
        def f(x, a):
            return a*x

        x = df["Eff 24h Tac Dose"][day_num-2:day_num].to_numpy()
        y = df["Tac level (prior to am dose)"][day_num-2:day_num].to_numpy()
        popt, pcov = curve_fit(f, x, y)

        fittedParameters = [popt[0], 0]

        # Calculate prediction based on linear fit
        prediction = np.polyval(fittedParameters, df["Eff 24h Tac Dose"][day_num])

        # Calculate deviation from prediction
        deviation = prediction - df["Tac level (prior to am dose)"][day_num]
        abs_deviation = abs(deviation)

        # Add prediction into dataframe
        df_L_RW_origin_int_day = np.array([pred_day, fittedParameters[0], fittedParameters[1], prediction, deviation, abs_deviation])
        df_L_RW_origin_int_day = pd.DataFrame(df_L_RW_origin_int_day.reshape(-1, len(df_L_RW_origin_int_day)),columns=column_names)
        df_L_RW_origin_int = df_L_RW_origin_int.append(df_L_RW_origin_int_day)

    df_L_RW_origin_int = df_L_RW_origin_int.reset_index(drop = True)
    return df_L_RW_origin_int

def prep_Cum_origin_dp_data(cal_pred_dataframe, index_first_prediction, deg):
    """
    Insert row with 0 tac dose and 0 tac level as origin data point and prepare prediction-ready dataframe for 
    Cum_origin_dp methods (both linear and quadratic).
    
    Input:
        cal_pred_dataframe: calibration and prediction dataframe, cal_pred[patient]
        index_first_prediction: index after inserting row for origin (3 for linear, 4 for quadratic)
        deg: degree of fitting polynomial (1 for linear, 2 for quadratic)
        
    Output: 
        Prediction-ready dataframe for Cum_origin_dp method
    """

    max_cum_length = len(cal_pred_dataframe)
    col_names = ['Pred_Day'] + ['Dose_' + str(i) for i in range(1, max_cum_length + 1)] + \
                ['Response_' + str(i) for i in range(1, max_cum_length + 1)] + \
                ['New_Dose', 'New_Response']
    prediction_dataframe = pd.DataFrame(columns = col_names)

    # Insert origin row
    cal_pred_dataframe.loc[-1] = [-1, 0, 0]  # add a row
    cal_pred_dataframe.index = cal_pred_dataframe.index + 1  # shift index
    cal_pred_dataframe.sort_index(inplace=True)

    # Prepare dataframe for prediction:
    # Loop through rows starting from first prediction
    for i in range(index_first_prediction, len(cal_pred_dataframe)):

        # Find prediction day, doses, and responses for each prediction
        pred_day = cal_pred_dataframe['Day #'][i]
        dose = cal_pred_dataframe['Eff 24h Tac Dose'][0: i].tolist()
        response = cal_pred_dataframe['Tac level (prior to am dose)'][0: i].tolist()
        new_dose = cal_pred_dataframe['Eff 24h Tac Dose'][i]
        new_response = cal_pred_dataframe['Tac level (prior to am dose)'][i]

        # Append results to prediction dataframe
        each_prediction = pd.DataFrame(columns = col_names)
        each_prediction.loc[0, 'Pred_Day'] = pred_day
        each_prediction.loc[0, 'Dose_1':'Dose_' + str(i)] = dose
        each_prediction.loc[0, 'Response_1':'Response_' + str(i)] = response
        each_prediction.loc[0, 'New_Dose'] = new_dose
        each_prediction.loc[0, 'New_Response'] = new_response
        prediction_dataframe = pd.concat([prediction_dataframe, each_prediction])

    prediction_dataframe = prediction_dataframe.reset_index(drop=True)
    
    return prediction_dataframe

def Cum_origin_dp(df_Cum_origin_dp, patient, prediction_dataframe, min_num_of_pairs, deg):
    """
    Use Cum_origin_dp method to generate predictions and calculate deviations.
    
    Input: 
        df_Cum_origin_dp: dictionary of dataframes for this method (df_Cum_origin_dp)
        patient: string of patient number
        prediction_dataframe: prediction-ready dataframe for a patient (prediction_dataframe[patient])
        min_num_of_pairs: minimum number of dose-response pairs including origin to start prediction
                          (3 for linear, 4 for quadratic)
        deg: degree of fitting polynomial (1 for linear, 2 for quadratic)
        
    Output: 
        Dataframe of results of method for a patient
    """
    # Perform Cum_origin_dp method:
    # Define dataframe output for Cum method
    df_Cum_origin_dp[patient] = pd.DataFrame(columns = ['prediction day', 'a', 'b', 'c', 'prediction', 'deviation', 'abs deviation'])
    max_cum_length = len(prediction_dataframe) + min_num_of_pairs - 1
    
    # Loop through rows
    for i in range(0, len(prediction_dataframe)):

        # Fit equation
        x = prediction_dataframe.loc[i, 'Dose_1':'Dose_' + str(max_cum_length)].dropna().astype(float)
        y = prediction_dataframe.loc[i, 'Response_1':'Response_' + str(max_cum_length)].dropna().astype(float)
        fittedParameters = (np.polyfit(x, y, deg))

        # Predict new response
        prediction = np.polyval(fittedParameters, prediction_dataframe.loc[i, 'New_Dose'])

        # Calculate deviation
        deviation = prediction - prediction_dataframe.loc[i, 'New_Response']
        abs_deviation = abs(deviation)

        # Append results into dataframe
        if deg == 2:
            dict = {'prediction day': prediction_dataframe.loc[i, 'Pred_Day'],
                'a': fittedParameters[0],
                'b': fittedParameters[1],
                'c': fittedParameters[2],
                'prediction': prediction,
                'deviation': deviation,
                'abs deviation': abs_deviation}
        else:
            dict = {'prediction day': prediction_dataframe.loc[i, 'Pred_Day'],
                'a': fittedParameters[0],
                'b': fittedParameters[1],
                'prediction': prediction,
                'deviation': deviation,
                'abs deviation': abs_deviation}

        df_Cum_origin_dp[patient] = df_Cum_origin_dp[patient].append(dict, ignore_index=True)

    return df_Cum_origin_dp[patient]

def PPM_origin_dp(cal_pred_dataframe, deg, df_PPM_origin_dp, patient):
    """
    Use PPM method with origin inserted as the first data point to generate predictions and calculate deviations.
    
    Input:
        cal_pred_dataframe: calibration and prediction data (linear_cal_pred[patient] or quad_cal_pred[patient])
        deg: degree of fitting polynomial (1 for linear, 2 for quadratic)
        df_PPM_origin_dp: dictionary (as it is)
        patient: string of patient (as it is)
        
    Output:
        Results of PPM_origin_dp method
    """

    # 1. Fit equation for the first prediction

    # Create output dataframe
    col_names = ['prediction day', 'a', 'b', 'c', 'new_dose', 'new_response', 'prediction', 'deviation', 'abs deviation']
    df_PPM_origin_dp[patient] = pd.DataFrame(columns = col_names)

    # Fit equation for first prediction
    x = cal_pred_dataframe['Eff 24h Tac Dose'][0:deg+2].astype(float)
    y = cal_pred_dataframe['Tac level (prior to am dose)'][0:deg+2].astype(float)
    fittedParameters = (np.polyfit(x, y, deg))

    # Add results to output dataframe:
    # Fill in prediction day, fittedParameters, new_dose and new_response
    dict = {'prediction day': cal_pred_dataframe['Day #'][deg+2],
           'a': fittedParameters[0],
           'b': fittedParameters[1],
           'new_dose': cal_pred_dataframe['Eff 24h Tac Dose'][deg+2],
           'new_response': cal_pred_dataframe['Tac level (prior to am dose)'][deg+2]}
    if deg == 2: # If quadratic, add 'c' to dict
        dict['c'] = fittedParameters[2]

    # Append results to dataframe
    df_PPM_origin_dp[patient] = df_PPM_origin_dp[patient].append(dict, ignore_index=True)

    # Predict new response
    prediction = np.polyval(fittedParameters, df_PPM_origin_dp[patient].loc[0, 'new_dose'])

    # Calculate deviations
    deviation = prediction - df_PPM_origin_dp[patient].loc[0, 'new_response']
    abs_deviation = abs(deviation)

    # Append prediction and deviations to dataframe
    df_PPM_origin_dp[patient].loc[0, 'prediction':'abs deviation'] = [prediction, deviation, abs_deviation]

    # 2. Fill in data for subsequent predictions

    # Loop through rows after data for first prediction
    for i in range(1, len(cal_pred_dataframe) - (deg + 2)):

        if deg == 2:

            # Find new c
            c = (df_PPM_origin_dp[patient].loc[i-1, 'c']) - (df_PPM_origin_dp[patient].loc[i-1, 'deviation'])

            # Create dict of prediction day, a, b, c, new_dose, new_response
            dict = {'prediction day': df_PPM_origin_dp[patient].loc[i-1,'prediction day'] + 1,
                   'a': df_PPM_origin_dp[patient].loc[i-1,'a'],
                   'b': df_PPM_origin_dp[patient].loc[i-1,'b'],
                   'c': c,
                   'new_dose': cal_pred_dataframe['Eff 24h Tac Dose'][deg+2+i],
                   'new_response': cal_pred_dataframe['Tac level (prior to am dose)'][deg+2+i]}

            # Append results to dataframe
            df_PPM_origin_dp[patient] = df_PPM_origin_dp[patient].append(dict, ignore_index=True)

            # Predict response
            prediction = np.polyval([df_PPM_origin_dp[patient].loc[i, 'a'], 
                                     df_PPM_origin_dp[patient].loc[i, 'b'],
                                     df_PPM_origin_dp[patient].loc[i, 'c']], 
                                     df_PPM_origin_dp[patient].loc[i, 'new_dose'])

            # Calculate deviations
            deviation = prediction - df_PPM_origin_dp[patient].loc[i, 'new_response']
            abs_deviation = abs(deviation)

            # Append prediction and deviations to dataframe
            df_PPM_origin_dp[patient].loc[i, 'prediction':'abs deviation'] = [prediction, deviation, abs_deviation]
        
        else: # Repeat all with one less fitted parameter
            # Find new b
            b = (df_PPM_origin_dp[patient].loc[i-1, 'b']) - (df_PPM_origin_dp[patient].loc[i-1, 'deviation'])

            # Create dict of prediction day, a, b, new_dose, new_response
            dict = {'prediction day': df_PPM_origin_dp[patient].loc[i-1,'prediction day'] + 1,
                   'a': df_PPM_origin_dp[patient].loc[i-1,'a'],
                   'b': b,
                   'new_dose': cal_pred_dataframe['Eff 24h Tac Dose'][deg+2+i],
                   'new_response': cal_pred_dataframe['Tac level (prior to am dose)'][deg+2+i]}

            # Append results to dataframe
            df_PPM_origin_dp[patient] = df_PPM_origin_dp[patient].append(dict, ignore_index=True)

            # Predict response
            prediction = np.polyval([df_PPM_origin_dp[patient].loc[i, 'a'], 
                                     df_PPM_origin_dp[patient].loc[i, 'b']],
                                     df_PPM_origin_dp[patient].loc[i, 'new_dose'])

            # Calculate deviations
            deviation = prediction - df_PPM_origin_dp[patient].loc[i, 'new_response']
            abs_deviation = abs(deviation)

            # Append prediction and deviations to dataframe
            df_PPM_origin_dp[patient].loc[i, 'prediction':'abs deviation'] = [prediction, deviation, abs_deviation]
            
    return df_PPM_origin_dp[patient]

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

def RMSE_MAE_without_intercept(df_deviation):
    # Plot RMSE and MAE
    df_rmse_MAE = pd.DataFrame()

    ## Plot RMSE
    methods = ['Q_Cum', 'Q_PPM', 'Q_RW', 'L_Cum', 'L_PPM', 'L_RW']

    rmse_Q_Cum = np.sqrt(mean(df_deviation['Q_Cum']**2))
    rmse_Q_PPM = np.sqrt(mean(df_deviation['Q_PPM']**2))
    rmse_Q_RW = np.sqrt(mean(df_deviation['Q_RW']**2))
    rmse_L_Cum = np.sqrt(mean(df_deviation['L_Cum']**2))
    rmse_L_PPM = np.sqrt(mean(df_deviation['Q_Cum']**2))
    rmse_L_RW = np.sqrt(mean(df_deviation['L_RW']**2))

    rmse = np.array([rmse_Q_Cum, rmse_Q_PPM, rmse_Q_RW, 
                    rmse_L_Cum, rmse_L_PPM, rmse_L_RW])

    rmse = pd.DataFrame(rmse.reshape(-1, len(rmse)),columns=methods)
    rmse=rmse.transpose()

    ## Calculate MAE
    MAE_Q_Cum = mean(abs(df_deviation['Q_Cum']))
    MAE_Q_PPM = mean(abs(df_deviation['Q_PPM']))
    MAE_Q_RW = mean(abs(df_deviation['Q_RW']))
    MAE_L_Cum = mean(abs(df_deviation['L_Cum']))
    MAE_L_PPM = mean(abs(df_deviation['L_PPM']))
    MAE_L_RW = mean(abs(df_deviation['L_RW']))

    MAE = np.array([MAE_Q_Cum, MAE_Q_PPM, MAE_Q_RW, 
                    MAE_L_Cum, MAE_L_PPM, MAE_L_RW])

    MAE = pd.DataFrame(MAE.reshape(-1, len(MAE)),columns=methods)
    MAE=MAE.transpose()

    df_rmse_MAE = df_rmse_MAE.append(rmse)
    df_rmse_MAE = pd.concat([df_rmse_MAE, MAE], axis=1)
    df_rmse_MAE.columns = ['RMSE', 'MAE']

    df_rmse_MAE.plot()
    plt.ylabel('RMSE and MAE')
    plt.title("RMSE and MAE of Deviation of Predicted from Actual Value")
    # plt.savefig('RMSE and MAE Deviation.png', bbox_inches="tight", dpi=300)

def tac_level_within_range(df):
    # Plot tac level over number of days from day 5
    sub_df = df.iloc[3:,]

    plt.scatter(x = sub_df["Day #"], 
                y = [sub_df["Tac level (prior to am dose)"]])
    plt.axhline(y=8, color='r', linestyle='-')
    plt.axhline(y=10, color='r', linestyle='-')

    df = (sub_df["Tac level (prior to am dose)"] > 8) & \
    (sub_df["Tac level (prior to am dose)"] < 10)

    perc_in_range = str(df.sum()) + " / " + str(sub_df["Tac level (prior to am dose)"].count()) + \
    " x 100% = " + str(round(df.sum()/sub_df["Tac level (prior to am dose)"].count()*100, 1)) + "%"
    perc_in_range

    plt.title("Tac Level within Range")
    plt.ylabel("Tac Level")
    plt.xlabel("No. of Days")
    plt.figtext(0.5, -0.1, "Percentage of tac level within range (from Day 5):\n" + perc_in_range, wrap=True, horizontalalignment='center', fontsize=12)

    # plt.savefig('Tac Level within Range.png', bbox_inches="tight", dpi=300)

def tac_level_within_range_fr_day_4(df):
    # Plot tac level over number of days from day 4
    sub_df = df.iloc[2:,]

    plt.scatter(x = sub_df["Day #"], 
                y = [sub_df["Tac level (prior to am dose)"]])
    plt.axhline(y=8, color='r', linestyle='-')
    plt.axhline(y=10, color='r', linestyle='-')

    df = (sub_df["Tac level (prior to am dose)"] > 8) & \
    (sub_df["Tac level (prior to am dose)"] < 10)

    perc_in_range = str(df.sum()) + " / " + str(sub_df["Tac level (prior to am dose)"].count()) + \
    " x 100% = " + str(round(df.sum()/sub_df["Tac level (prior to am dose)"].count()*100, 1)) + "%"
    perc_in_range

    plt.title("Tac Level within Range")
    plt.ylabel("Tac Level")
    plt.xlabel("No. of Days")
    plt.figtext(0.5, -0.1, "Percentage of tac level within range (from Day 4):\n" + perc_in_range, wrap=True, horizontalalignment='center', fontsize=12)

    # plt.savefig('Tac Level within Range_from Day 4.png', bbox_inches="tight", dpi=300)