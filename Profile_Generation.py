from openpyxl import load_workbook
import pandas as pd

def get_sheet_names(input_file):
    """ Get sheet names which are also patient names """
    wb = load_workbook(input_file, read_only=True)
    patient_list = wb.sheetnames
    wb.close()
    return patient_list

def clean_data(df, patient):
    """ 
    Clean data. Returns cleaned dataframe.
    
    Details:
    Keep target columns from excel, shift tac level one cell up, remove "mg"/"ng" 
    from dose, replace NaN with 0 in dose.
    
    Input:
    df - dataframe from each sheet
    patient - patient number from list of patients
    
    Output:
    df - cleaned dataframe        
    """

    # Keep target columns
    df = df[["Day #", "Tac level (prior to am dose)", "Eff 24h Tac Dose"]]

    # Shift tac level one cell up to match dose-response to one day
    df['Tac level (prior to am dose)'] = df['Tac level (prior to am dose)'].shift(-1)

    # Remove "mg"/"ng" from dose
    df['Eff 24h Tac Dose'] = df['Eff 24h Tac Dose'].astype(str).str.replace('mg', '')
    df['Eff 24h Tac Dose'] = df['Eff 24h Tac Dose'].astype(str).str.replace('ng', '')
    df['Eff 24h Tac Dose'] = df['Eff 24h Tac Dose'].astype(float)

    # Replace NA with 0 in dose column
    df["Eff 24h Tac Dose"] = df["Eff 24h Tac Dose"].fillna(0)
    
    return df

def keep_ideal_data(df, patient, list_of_patient_df):
    """
    Remove rows with non-ideal data, including missing tac level and/or tac dose, 
    <2 tac level, multiple blood draws. Then keep the longest consecutive chunk
    based on number of days. 

    Input: Dataframe of individual patient
    Output: Dataframe with the longest chunk of consecutive ideal data.
    """
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
        df_temp = print("there are >1 chunks of data with the longest length.")
    else:
        # Find index of largest chunk to keep in dataframe
        min_idx = df_temp.loc[0, 'min'] + 1 # First index where non-NA begins is 1 after the min index, thus add 1 to min index
        max_idx = df_temp.loc[0, 'max'] # Get max index where non-NA chunk ends

        # Keep largest chunk in dataframe
        df = df.iloc[min_idx:max_idx + 1, :] 

    # Format patient dataframe
    df['patient'] = patient
    df.columns = ['day', 'response', 'dose', 'patient']
    list_of_patient_df.append(df)

    return df

def cal_pred_data(df, patient, patients_to_exclude, deg):
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
    df = df.reset_index(drop=False) 

    # Create boolean column to check if tac dose diff from next row
    df['diff_from_next'] = \
            (df['dose'] != df['dose'].shift(-1))

    # Find indexes of last rows of first 2 unique doses
    last_unique_doses_idx = [i for i, x in enumerate(df.diff_from_next) if x]

    # Create boolean column to check if tac dose diff from previous row
    df['diff_from_prev'] = \
            (df['dose'] != df['dose'].shift(1))

    # Find indexes of first rows of third unique dose
    first_unique_doses_idx = [i for i, x in enumerate(df.diff_from_prev) if x]

    # The boolean checks created useless index, diff_from_next and diff_from_prev columns,
    # drop them
    df = df.drop(['index', 'diff_from_next', 'diff_from_prev'], axis=1)

    # Combine calibration and prediction rows
    cal_pred = pd.DataFrame()

    # Do for quadratic method
    if deg == 2:
        if df['dose'].nunique() > 2:
            # If third cal point is the same as first 2 cal points, keep looking
            first_cal_dose = df['dose'][last_unique_doses_idx[0]]
            second_cal_dose = df['dose'][last_unique_doses_idx[1]]
            n = 2
            for i in range(n, len(df)+1):
                third_cal_dose = df['dose'][first_unique_doses_idx[n]]
                if (third_cal_dose == first_cal_dose) | (third_cal_dose == second_cal_dose):
                    n = n + 1

            first_cal_point = pd.DataFrame(df.iloc[last_unique_doses_idx[0],:]).T
            second_cal_point = pd.DataFrame(df.iloc[last_unique_doses_idx[1],:]).T
            third_cal_point = pd.DataFrame(df.iloc[first_unique_doses_idx[n],:]).T
            rest_of_data = df.iloc[first_unique_doses_idx[n]+1:,:]
            cal_pred = pd.concat([first_cal_point, second_cal_point, third_cal_point, 
                                        rest_of_data]).reset_index(drop=True)
        else:
            patients_to_exclude.append(str(patient))
            print(f"Patient #{patient} has insufficient unique dose-response pairs for calibration (for quad)!")
            
    # Do for linear method
    if deg == 1:
        if df['dose'].nunique() > 1:
            first_cal_point = pd.DataFrame(df.iloc[last_unique_doses_idx[0],:]).T
            second_cal_point = pd.DataFrame(df.iloc[first_unique_doses_idx[1],:]).T
            rest_of_data = df.iloc[first_unique_doses_idx[1]+1:,:]
            cal_pred = pd.concat([first_cal_point, second_cal_point, 
                                        rest_of_data]).reset_index(drop=True)
        else:
            patients_to_exclude.append(str(patient))
            print(f"Patient #{patient} has insufficient unique dose-response pairs for calibration (for linear)!")

    # Print error msg if number of predictions is less than 3
    if (len(cal_pred) - (deg + 1) < 3) and (df['dose'].nunique() > deg):
        patients_to_exclude.append(str(patient))
        if deg == 1:
            error_string = '(for linear)'
        else:
            error_string = '(for quadratic)'
            
        print(f"Patient #{patient} has insufficient/<3 predictions ({len(cal_pred) - (deg + 1)} predictions) {error_string}!")

    # Add "type" column
    cal_pred['type'] = ""
    if deg == 1:
        cal_pred['type'] = 'linear'
    else:
        cal_pred['type'] = 'quadratic'

    return cal_pred, patients_to_exclude

def keep_target_patients(patient, patients_to_exclude_linear, patients_to_exclude_quad, cal_pred_linear, cal_pred_quad, \
                        list_of_cal_pred_df):
    """
    Keep patient data with sufficient dose-response pairs and predictions
    
    Details:
    There are 4 scenarios depending on inclusion/exlusion of patient for linear/quad methods.
    Append patient data to list_of_cal_pred_df only if there's sufficient dose-response pairs and predictions.
    
    Input:
    patient
    patients_to_exclude_linear - list of patients to be excluded from linear methods
    patients_to_exclude_quad - list of patients to be excluded from quad methods
    cal_pred_linear - individual patient dataframe with calibration and efficacy-driven dosing data for linear methods
    cal_pred_quad - same as above, for quad methods
    list_of_cal_pred_df - list of all cal_pred dataframes of each patient from first to current patient in loop
    
    Output:
    cal_pred
        - individual patient dataframe of calibration and efficacy-driven dosing, 
          that has sufficient dose-response pairs and predictions
    list_of_cal_pred_df 
        - list of all cal_pred dataframes of each patient 
          from first to current patient in loop
    """
    if (patient not in patients_to_exclude_linear) and (patient not in patients_to_exclude_quad):
        cal_pred = pd.concat([cal_pred_linear, cal_pred_quad])
        list_of_cal_pred_df.append(cal_pred)
    elif patient in patients_to_exclude_linear and (patient not in patients_to_exclude_quad):
        cal_pred = cal_pred_quad
        list_of_cal_pred_df.append(cal_pred)
    elif patient not in patients_to_exclude_linear and (patient in patients_to_exclude_quad):
        cal_pred = cal_pred_linear
        list_of_cal_pred_df.append(cal_pred)
    else: cal_pred = pd.DataFrame()
    return cal_pred, list_of_cal_pred_df