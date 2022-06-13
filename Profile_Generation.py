from openpyxl import load_workbook
import pandas as pd
from scipy.optimize import curve_fit
import numpy as np
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

##### MAIN FUNCTIONS #####
# Generate profiles
def generate_profiles():
    """
    Generate profiles for patients.
    
    Details:
    Define parameters and lists. Loop through patients to clean data, select data for calibration and efficacy-driven dosing, keep patients with
    sufficient dose-response pairs and predictions for each method, then apply method.
    
    Outputs:
    - patients_to_exclude_linear: list of patients to exclude for linear methods.
    - patients_to_exclude_quad: list of patients to exclude for quadratic methods.
    - list_of_patient_df: list of dataframes of cleaned patient data.
    - list_of_cal_pred_df: list of dataframes of data for calibration and efficacy-driven dosing, that has sufficient dose-response pairs and predictions
    - list_of_result_df: list of dataframes with results after applying methods.
    """

    # Profile Generation
    input_file = 'Retrospective Liver Transplant Data.xlsx'
    rows_to_skip = 17

    # Get list of patients/sheet names
    list_of_patients = get_sheet_names(input_file)

    # Define lists
    list_of_patient_df = []
    list_of_cal_pred_df = []
    list_of_result_df = []
    patients_to_exclude_linear = []
    patients_to_exclude_quad = []

    for patient in list_of_patients:

        # Create and clean patient dataframe        
        df = pd.read_excel(input_file, sheet_name=patient, skiprows=rows_to_skip)
        df = clean_data(df, patient)
        df = keep_ideal_data(df, patient, list_of_patient_df)

        # Select data for calibration and efficacy-driven dosing
        cal_pred_linear, patients_to_exclude_linear = cal_pred_data(df, patient, patients_to_exclude_linear, 1)
        cal_pred_quad, patients_to_exclude_quad = cal_pred_data(df, patient, patients_to_exclude_quad, 2)

        # Keep patients with sufficient dose-response pairs and predictions for each method
        cal_pred, list_of_cal_pred_df = keep_target_patients(patient, patients_to_exclude_linear, patients_to_exclude_quad, 
                                                         cal_pred_linear, cal_pred_quad, list_of_cal_pred_df)   

        # Apply methods
        list_of_result_df = apply_methods(cal_pred, patient, patients_to_exclude_linear, patients_to_exclude_quad,
                  cal_pred_linear, cal_pred_quad, list_of_result_df)
        
    return patients_to_exclude_linear, patients_to_exclude_quad, list_of_patient_df, list_of_cal_pred_df, list_of_result_df

def print_patients_to_exclude(patients_to_exclude_linear, patients_to_exclude_quad):
    """Print patients to exclude"""        
    patients_to_exclude_linear = sorted(set(patients_to_exclude_linear))
    patients_to_exclude_quad = sorted(set(patients_to_exclude_quad))
    print(f"Patients to exclude for linear methods: {patients_to_exclude_linear}")
    print(f"Patients to exclude for quad methods: {patients_to_exclude_quad}")

def join_dataframes(list_of_patient_df, list_of_cal_pred_df, list_of_result_df):
    """Join dataframes from individual patients"""
    df = pd.concat(list_of_patient_df)
    df.patient = df.patient.apply(int)
    df.reset_index(inplace=True, drop=True)
    cal_pred = pd.concat(list_of_cal_pred_df)
    cal_pred.patient = cal_pred.patient.apply(int)
    result_df = pd.concat(list_of_result_df)
    result_df = format_result_df(cal_pred, result_df)
    
    return df, cal_pred, result_df

def output_df_to_excel(df, cal_pred, result_df):
    """Output dataframes to excel as individual sheets"""
    with pd.ExcelWriter('output.xlsx') as writer:
        df.to_excel(writer, sheet_name='clean', index=False)
        cal_pred.to_excel(writer, sheet_name='calibration_and_efficacy_driven', index=False)
        result_df.to_excel(writer, sheet_name='result', index=False)

##### SUPPORTING FUNCTIONS ######
# Create patient dataframe
def get_sheet_names(input_file):
    """ Get sheet names which are also patient names """
    wb = load_workbook(input_file, read_only=True)
    patient_list = wb.sheetnames
    wb.close()
    return patient_list

def clean_data(df, patient):
    """ 
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

    # Set boolean for non_ideal as True if all dose including and above current row is 0
    for i in range(len(df_temp)):
        if (df_temp.loc[0:i, 'Eff 24h Tac Dose'] == 0).all():
            df_temp.loc[i, 'non_ideal'] = True

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

    cal_pred.reset_index(inplace=True, drop=True)

    return cal_pred, list_of_cal_pred_df

# Prepare patient dataframe for prediction and apply method
def apply_methods(cal_pred, patient, patients_to_exclude_linear, patients_to_exclude_quad,
                  cal_pred_linear, cal_pred_quad, list_of_result_df):
    
    """
    If cal_pred is filled, create result dataframe and apply all methods.
    
    Input:
    cal_pred - combined dataframe of cal_pred_linear and cal_pred_quad that satisfy minimum criteria.
    patient
    patients_to_exclude_linear - list of patients to exclude for linear methods
    patients_to_exclude_linear - list of patients to exclude for quadratic methods
    cal_pred_linear - individual patient dataframe with calibration and efficacy-driven dosing data for linear methods
    cal_pred_quad - same as above, for quad methods
    list_of_result_df - list of result dataframe from each patient
    
    Output:
    list_of_result_df - list of result dataframe from each patient
    """
    
    if len(cal_pred) != 0:

        # Create result dataFrame
        max_count_input = len(cal_pred_linear)

        col_names = ['patient', 'method', 'pred_day'] + \
                    ['fit_dose_' + str(i) for i in range(1, max_count_input + 1)] + \
                    ['fit_response_' + str(i) for i in range(1, max_count_input + 1)] + \
                    ['day_' + str(i) for i in range(1, max_count_input + 1)] + \
                    ['weight_' + str(i) for i in range(1, max_count_input + 1)] + \
                    ['half_life', 'dose', 'response', 'prev_coeff_2x', 'prev_coeff_1x', 'prev_coeff_0x',\
                     'prev_deviation', 'coeff_2x', 'coeff_1x', 'coeff_0x', 'prediction', 'deviation',
                     'abs_deviation']
        result = pd.DataFrame(columns=col_names)    

        if patient not in patients_to_exclude_linear:
            deg = 1
            list_of_result_df = Cum(deg, cal_pred_linear, result, 'L_Cum_wo_origin', list_of_result_df, 'wo_origin', tau="")
            list_of_result_df = Cum(deg, cal_pred_linear, result, 'L_Cum_wo_origin_tau', list_of_result_df, 'wo_origin', tau=1)
            list_of_result_df = Cum(deg, cal_pred_linear, result, 'L_Cum_origin_dp', list_of_result_df, 'origin_dp', tau="")
            list_of_result_df = Cum(deg, cal_pred_linear, result, 'L_Cum_origin_dp_tau', list_of_result_df, 'origin_dp', tau=1)
            list_of_result_df = Cum(deg, cal_pred_linear, result, 'L_Cum_origin_int', list_of_result_df, 'origin_int', tau="")
            list_of_result_df = Cum(deg, cal_pred_linear, result, 'L_Cum_origin_int_tau', list_of_result_df, 'origin_int', tau=1)
            list_of_result_df = PPM(deg, cal_pred_linear, result, 'L_PPM_wo_origin', list_of_result_df, 'wo_origin', tau="")
            list_of_result_df = PPM(deg, cal_pred_linear, result, 'L_PPM_wo_origin_tau', list_of_result_df, 'wo_origin', tau=1)
            list_of_result_df = PPM(deg, cal_pred_linear, result, 'L_PPM_origin_dp', list_of_result_df, 'origin_dp', tau="")
            list_of_result_df = PPM(deg, cal_pred_linear, result, 'L_PPM_origin_dp_tau', list_of_result_df, 'origin_dp', tau=1)
            list_of_result_df = PPM(deg, cal_pred_linear, result, 'L_PPM_origin_int', list_of_result_df, 'origin_int', tau="")
            list_of_result_df = PPM(deg, cal_pred_linear, result, 'L_PPM_origin_int_tau', list_of_result_df, 'origin_int', tau=1)
            list_of_result_df = RW(deg, cal_pred_linear, result, 'L_RW_wo_origin', list_of_result_df, 'wo_origin', tau="")
            list_of_result_df = RW(deg, cal_pred_linear, result, 'L_RW_wo_origin_tau', list_of_result_df, 'wo_origin', tau=1)
            list_of_result_df = RW(deg, cal_pred_linear, result, 'L_RW_origin_dp', list_of_result_df, 'origin_dp', tau="")
            list_of_result_df = RW(deg, cal_pred_linear, result, 'L_RW_origin_dp_tau', list_of_result_df, 'origin_dp', tau=1)
            list_of_result_df = RW(deg, cal_pred_linear, result, 'L_RW_origin_int', list_of_result_df, 'origin_int', tau="")
            list_of_result_df = RW(deg, cal_pred_linear, result, 'L_RW_origin_int_tau', list_of_result_df, 'origin_int', tau=1)

        if patient not in patients_to_exclude_quad:
            deg = 2
            list_of_result_df = Cum(deg, cal_pred_quad, result, 'Q_Cum_wo_origin', list_of_result_df, 'wo_origin', tau="")
            list_of_result_df = Cum(deg, cal_pred_quad, result, 'Q_Cum_wo_origin_tau', list_of_result_df, 'wo_origin', tau=1)
            list_of_result_df = Cum(deg, cal_pred_quad, result, 'Q_Cum_origin_dp', list_of_result_df, 'origin_dp', tau="")
            list_of_result_df = Cum(deg, cal_pred_quad, result, 'Q_Cum_origin_dp_tau', list_of_result_df, 'origin_dp', tau=1)
            list_of_result_df = Cum(deg, cal_pred_quad, result, 'Q_Cum_origin_int', list_of_result_df, 'origin_int', tau="")
            list_of_result_df = Cum(deg, cal_pred_quad, result, 'Q_Cum_origin_int_tau', list_of_result_df, 'origin_int', tau=1)
            list_of_result_df = PPM(deg, cal_pred_quad, result, 'Q_PPM_wo_origin', list_of_result_df, 'wo_origin', tau="")
            list_of_result_df = PPM(deg, cal_pred_quad, result, 'Q_PPM_wo_origin_tau', list_of_result_df, 'wo_origin', tau=1)
            list_of_result_df = PPM(deg, cal_pred_quad, result, 'Q_PPM_origin_dp', list_of_result_df, 'origin_dp', tau="")
            list_of_result_df = PPM(deg, cal_pred_quad, result, 'Q_PPM_origin_dp_tau', list_of_result_df, 'origin_dp', tau=1)
            list_of_result_df = PPM(deg, cal_pred_quad, result, 'Q_PPM_origin_int', list_of_result_df, 'origin_int', tau="")
            list_of_result_df = PPM(deg, cal_pred_quad, result, 'Q_PPM_origin_int_tau', list_of_result_df, 'origin_int', tau=1)
            list_of_result_df = RW(deg, cal_pred_quad, result, 'Q_RW_wo_origin', list_of_result_df, 'wo_origin', tau="")
            list_of_result_df = RW(deg, cal_pred_quad, result, 'Q_RW_wo_origin_tau', list_of_result_df, 'wo_origin', tau=1)
            list_of_result_df = RW(deg, cal_pred_quad, result, 'Q_RW_origin_dp', list_of_result_df, 'origin_dp', tau="")
            list_of_result_df = RW(deg, cal_pred_quad, result, 'Q_RW_origin_dp_tau', list_of_result_df, 'origin_dp', tau=1)
            list_of_result_df = RW(deg, cal_pred_quad, result, 'Q_RW_origin_int', list_of_result_df, 'origin_int', tau="")
            list_of_result_df = RW(deg, cal_pred_quad, result, 'Q_RW_origin_int_tau', list_of_result_df, 'origin_int', tau=1)

    return list_of_result_df

def linear_func(x, a, b):
    return a * x + b

def linear_func_origin_int(x, a, b):
    return a * x + 0

def quad_func(x, a, b, c):
    return a * (x ** 2) + b * x + c

def quad_func_origin_int(x, a, b, c):
    return a * (x ** 2) + b * x + 0

def Cum(deg, cal_pred, result, method_string, list_of_result_df, origin_inclusion='wo_origin', tau=""):
    """
    Execute cumulative approach for each variation of origin inclusion, type, and tau inclusion. 

    Details:
    (If the chosen method has tau, the following details are looped for each variation of half_life)
    Add a fresh line to result dataframe for each prediction and fill in the line with patient, method,
    prediction day (and half-life).Fill in the dose-response pairs and days of the pairs that are 
    required to fit the model, according to the origin inclusion chosen. Fill in the dose-response 
    pair of the day of the prediction to calculate the deviation of the prediction. (If chosen method 
    has tau, calculate the weights based on the number of days from prediction day). Fit the model 
    (with weight if chosen method has tau). Calculate the prediction, deviation, and absolute deviation, 
    and fill in the results to the result dataframe.

    Input:
    deg - degree of polynomial fit, 1 for linear, 2 for quadratic
    cal_pred - dataframe of calibration and efficacy-driven dosing data, 
                cal_pred_linear for linear, cal_pred_quad for quad
    result - dataframe of results
    method_string - string of method
    list_of_result_df - list of result dataframe from each patient

    Output:
    list_of_result_df
    """
    j = 0

    result = result[0:0]
    
    half_life = np.arange(3.5, 41.5, 1)

    if tau == 1:

        # Loop through all half_lives
        for k in range(len(half_life)):

            result = result[0:0]

            # Loop through each prediction
            for i in range(deg + 1, len(cal_pred)):
            
                result.loc[j, 'patient'] = cal_pred.loc[i, 'patient']
                result.loc[j, 'method'] = method_string
                result.loc[j, 'pred_day'] = cal_pred.loc[i, 'day']
                if (origin_inclusion == 'wo_origin') or (origin_inclusion =='origin_int'):
                    result.loc[j, 'fit_dose_1':'fit_dose_' + str(i)] = cal_pred.loc[0:i-1, 'dose'].to_numpy()
                    result.loc[j, 'fit_response_1':'fit_response_' + str(i)] = cal_pred.loc[0:i-1, 'response'].to_numpy()
                    result.loc[j, 'day_1':'day_' + str(i)] = cal_pred.loc[0:i-1, 'day'].to_numpy()
                elif origin_inclusion == 'origin_dp':
                    result.loc[j, 'fit_dose_1'] = 0
                    result.loc[j, 'fit_dose_2':'fit_dose_' + str(i+1)] = cal_pred.loc[0:i-1, 'dose'].to_numpy()
                    result.loc[j, 'fit_response_1'] = 0
                    result.loc[j, 'fit_response_2':'fit_response_' + str(i+1)] = cal_pred.loc[0:i-1, 'response'].to_numpy()
                    result.loc[j, 'day_1'] = cal_pred.loc[0, 'day'] - 1
                    result.loc[j, 'day_2':'day_' + str(i+1)] = cal_pred.loc[0:i-1, 'day'].to_numpy()
                
                result.loc[j, 'dose'] = cal_pred.loc[i, 'dose']
                result.loc[j, 'response'] = cal_pred.loc[i, 'response']
                result.loc[j, 'half_life'] = half_life[k]

                # Calculate and fill in weight
                fixed_pred_day = result.loc[j, 'pred_day']
                fixed_half_life = result.loc[j, 'half_life']
                if (origin_inclusion == 'wo_origin') or (origin_inclusion =='origin_int'):
                    day_array = result.loc[j, 'day_1':'day_' + str(i)].astype(float)

                    # Fill in weight array
                    result.loc[j, 'weight_1':'weight_' + str(i)] = np.exp(-(24*(fixed_pred_day - day_array)*(math.log(2)/fixed_half_life))).to_numpy()

                elif origin_inclusion == 'origin_dp':
                    day_array = result.loc[j, 'day_1':'day_' + str(i+1)].astype(float)

                    # Fill in weight array
                    result.loc[j, 'weight_1':'weight_' + str(i+1)] = np.exp(-(24*(fixed_pred_day - day_array)*(math.log(2)/fixed_half_life))).to_numpy()

                # Curve fit equation
                if origin_inclusion == 'wo_origin' or origin_inclusion == 'origin_int':
                    X = result.loc[j, 'fit_dose_1':'fit_dose_' + str(i)].to_numpy()
                    y = result.loc[j, 'fit_response_1':'fit_response_' + str(i)].to_numpy()
                    weight = result.loc[j, 'weight_1':'weight_' + str(i)]
                elif origin_inclusion == 'origin_dp':
                    X = result.loc[j, 'fit_dose_1':'fit_dose_' + str(i+1)].to_numpy()
                    y = result.loc[j, 'fit_response_1':'fit_response_' + str(i+1)].to_numpy()
                    weight = result.loc[j, 'weight_1':'weight_' + str(i+1)]
                
                if deg == 1:
                    if origin_inclusion == 'origin_int':
                        poly_reg = PolynomialFeatures(degree=deg)
                        X = X.reshape(-1,1)
                        X = poly_reg.fit_transform(X)
                        fitted_model = LinearRegression(fit_intercept=True).fit(X, y, weight)
                        result.loc[j, 'coeff_0x'] = fitted_model.coef_[0]
                        result.loc[j, 'coeff_1x'] = fitted_model.coef_[1]
                    else:
                        poly_reg = PolynomialFeatures(degree=deg)
                        X = X.reshape(-1,1)
                        X = poly_reg.fit_transform(X)
                        fitted_model = LinearRegression(fit_intercept=False).fit(X, y, weight)
                        result.loc[j, 'coeff_0x'] = fitted_model.coef_[0]
                        result.loc[j, 'coeff_1x'] = fitted_model.coef_[1]
                else:
                    if origin_inclusion == 'origin_int':
                        poly_reg = PolynomialFeatures(degree=deg)
                        X = X.reshape(-1,1)
                        X = poly_reg.fit_transform(X)
                        fitted_model = LinearRegression(fit_intercept=True).fit(X, y, weight)
                        result.loc[j, 'coeff_0x'] = fitted_model.coef_[0]
                        result.loc[j, 'coeff_1x'] = fitted_model.coef_[1]
                        result.loc[j, 'coeff_2x'] = fitted_model.coef_[2]
                    else:
                        poly_reg = PolynomialFeatures(degree=deg)
                        X = X.reshape(-1,1)
                        X = poly_reg.fit_transform(X)
                        fitted_model = LinearRegression(fit_intercept=False).fit(X, y, weight)
                        result.loc[j, 'coeff_0x'] = fitted_model.coef_[0]
                        result.loc[j, 'coeff_1x'] = fitted_model.coef_[1]
                        result.loc[j, 'coeff_2x'] = fitted_model.coef_[2]

                # Calculate prediction and deviation
                
                prediction = fitted_model.predict(poly_reg.fit_transform([[cal_pred.loc[i, 'dose']]]))[0]

                result.loc[j, 'prediction'] = prediction
                deviation = cal_pred.loc[i, 'response'] - prediction
                result.loc[j, 'deviation'] = deviation
                abs_deviation = abs(deviation)
                result.loc[j, 'abs_deviation'] = abs_deviation
                
                j = j + 1

            list_of_result_df.append(result)

    else:

        result = result[0:0]

        # Loop through each prediction
        for i in range(deg + 1, len(cal_pred)):
    
            result.loc[j, 'patient'] = cal_pred.loc[i, 'patient']
            result.loc[j, 'method'] = method_string
            result.loc[j, 'pred_day'] = cal_pred.loc[i, 'day']
            if (origin_inclusion == 'wo_origin') or (origin_inclusion =='origin_int'):
                result.loc[j, 'fit_dose_1':'fit_dose_' + str(i)] = cal_pred.loc[0:i-1, 'dose'].to_numpy()
                result.loc[j, 'fit_response_1':'fit_response_' + str(i)] = cal_pred.loc[0:i-1, 'response'].to_numpy()
                result.loc[j, 'day_1':'day_' + str(i)] = cal_pred.loc[0:i-1, 'day'].to_numpy()
            elif origin_inclusion == 'origin_dp':
                result.loc[j, 'fit_dose_1'] = 0
                result.loc[j, 'fit_dose_2':'fit_dose_' + str(i+1)] = cal_pred.loc[0:i-1, 'dose'].to_numpy()
                result.loc[j, 'fit_response_1'] = 0
                result.loc[j, 'fit_response_2':'fit_response_' + str(i+1)] = cal_pred.loc[0:i-1, 'response'].to_numpy()
                result.loc[j, 'day_1'] = cal_pred.loc[0, 'day'] - 1
                result.loc[j, 'day_2':'day_' + str(i+1)] = cal_pred.loc[0:i-1, 'day'].to_numpy()
            
            result.loc[j, 'dose'] = cal_pred.loc[i, 'dose']
            result.loc[j, 'response'] = cal_pred.loc[i, 'response']

            # Curve fit equation
            if origin_inclusion == 'wo_origin' or origin_inclusion == 'origin_int':
                X = result.loc[j, 'fit_dose_1':'fit_dose_' + str(i)].to_numpy()
                y = result.loc[j, 'fit_response_1':'fit_response_' + str(i)].to_numpy()
            elif origin_inclusion == 'origin_dp':
                X = result.loc[j, 'fit_dose_1':'fit_dose_' + str(i+1)].to_numpy()
                y = result.loc[j, 'fit_response_1':'fit_response_' + str(i+1)].to_numpy()
            
            if deg == 1:
                if origin_inclusion == 'origin_int':
                    poly_reg = PolynomialFeatures(degree=deg)
                    X = X.reshape(-1,1)
                    X = poly_reg.fit_transform(X)
                    fitted_model = LinearRegression(fit_intercept=True).fit(X, y)
                    result.loc[j, 'coeff_0x'] = fitted_model.coef_[0]
                    result.loc[j, 'coeff_1x'] = fitted_model.coef_[1]
                else:
                    poly_reg = PolynomialFeatures(degree=deg)
                    X = X.reshape(-1,1)
                    X = poly_reg.fit_transform(X)
                    fitted_model = LinearRegression(fit_intercept=False).fit(X, y)
                    result.loc[j, 'coeff_0x'] = fitted_model.coef_[0]
                    result.loc[j, 'coeff_1x'] = fitted_model.coef_[1]
            else:
                if origin_inclusion == 'origin_int':
                    poly_reg = PolynomialFeatures(degree=deg)
                    X = X.reshape(-1,1)
                    X = poly_reg.fit_transform(X)
                    fitted_model = LinearRegression(fit_intercept=True).fit(X, y)
                    result.loc[j, 'coeff_0x'] = fitted_model.coef_[0]
                    result.loc[j, 'coeff_1x'] = fitted_model.coef_[1]
                    result.loc[j, 'coeff_2x'] = fitted_model.coef_[2]
                else:
                    poly_reg = PolynomialFeatures(degree=deg)
                    X = X.reshape(-1,1)
                    X = poly_reg.fit_transform(X)
                    fitted_model = LinearRegression(fit_intercept=False).fit(X, y)
                    result.loc[j, 'coeff_0x'] = fitted_model.coef_[0]
                    result.loc[j, 'coeff_1x'] = fitted_model.coef_[1]
                    result.loc[j, 'coeff_2x'] = fitted_model.coef_[2]
                
            # Calculate prediction and deviation
            prediction = fitted_model.predict(poly_reg.fit_transform([[cal_pred.loc[i, 'dose']]]))[0]

            result.loc[j, 'prediction'] = prediction
            deviation = cal_pred.loc[i, 'response'] - prediction
            result.loc[j, 'deviation'] = deviation
            abs_deviation = abs(deviation)
            result.loc[j, 'abs_deviation'] = abs_deviation
            
            j = j + 1

        list_of_result_df.append(result)

    return list_of_result_df

def PPM(deg, cal_pred, result, method_string, list_of_result_df, origin_inclusion='wo_origin', tau=""):
    """
    Execute PPM method. 

    Details:


    Prepare input dataframe for PPM_wo_origin method. 
    For first prediction, fill up dose and response to be fitted. Then fill in coefficients, prediction and deviation of fit.
    For second and following predictions, fill up dose and response of prediction day, fill in previous coefficients and deviations,
    shift last coefficient by previous deviation, fill in prediction and deviation of fit.

    Input:
    deg - degree of polynomial fit, 1 for linear, 2 for quadratic
    cal_pred - dataframe of calibration and efficacy-driven dosing data, 
                cal_pred_linear for linear, cal_pred_quad for quad
    result - dataframe of results
    method_string - string of method
    list_of_result_df - list of result dataframe from each patient

    Output:
    list_of_result_df
    """
    j = 0 # Counter for number of rows in "result"
        
    half_life = np.arange(3.5, 41.5, 1)

    if tau == 1:

        # Loop through all half_lives
        for k in range(len(half_life)):

            result = result[0:0]

            j = 0
                
            # Loop through each prediction
            for i in range(deg + 1, len(cal_pred)):

                result.loc[j, 'half_life'] = half_life[k]

                # First prediction
                if j == 0:                    
                    result.loc[j, 'patient'] = cal_pred.loc[i, 'patient']
                    result.loc[j, 'method'] = method_string
                    result.loc[j, 'pred_day'] = cal_pred.loc[i, 'day']

                    if (origin_inclusion == 'wo_origin') or (origin_inclusion == 'origin_int'):
                        result.loc[j, 'fit_dose_1':'fit_dose_' + str(i)] = cal_pred.loc[0:i-1, 'dose'].to_numpy()
                        result.loc[j, 'fit_response_1':'fit_response_' + str(i)] = cal_pred.loc[0:i-1, 'response'].to_numpy()
                        result.loc[j, 'day_1':'day_' + str(i)] = cal_pred.loc[0:i-1, 'day'].to_numpy()
                    elif origin_inclusion == 'origin_dp':
                        result.loc[j, 'fit_dose_1'] = 0
                        result.loc[j, 'fit_dose_2':'fit_dose_' + str(i+1)] = cal_pred.loc[0:i-1, 'dose'].to_numpy()
                        result.loc[j, 'fit_response_1'] = 0
                        result.loc[j, 'fit_response_2':'fit_response_' + str(i+1)] = cal_pred.loc[0:i-1, 'response'].to_numpy()
                        result.loc[j, 'day_1'] = cal_pred.loc[0, 'day'] - 1
                        result.loc[j, 'day_2':'day_' + str(i+1)] = cal_pred.loc[0:i-1, 'day'].to_numpy()

                    result.loc[j, 'dose'] = cal_pred.loc[i, 'dose']
                    result.loc[j, 'response'] = cal_pred.loc[i, 'response']
                    
                    # Calculate and fill in weight
                    fixed_pred_day = result.loc[j, 'pred_day']
                    fixed_half_life = result.loc[j, 'half_life']
                    if (origin_inclusion == 'wo_origin') or (origin_inclusion =='origin_int'):
                        day_array = result.loc[j, 'day_1':'day_' + str(i)].astype(float)

                        # Fill in weight array
                        result.loc[j, 'weight_1':'weight_' + str(i)] = np.exp(-(24*(fixed_pred_day - day_array)*(math.log(2)/fixed_half_life))).to_numpy()

                    elif origin_inclusion == 'origin_dp':
                        day_array = result.loc[j, 'day_1':'day_' + str(i+1)].astype(float)

                        # Fill in weight array
                        result.loc[j, 'weight_1':'weight_' + str(i+1)] = np.exp(-(24*(fixed_pred_day - day_array)*(math.log(2)/fixed_half_life))).to_numpy()

                    # Curve fit equation
                    if origin_inclusion == 'wo_origin' or origin_inclusion == 'origin_int':
                        X = result.loc[j, 'fit_dose_1':'fit_dose_' + str(i)].to_numpy()
                        y = result.loc[j, 'fit_response_1':'fit_response_' + str(i)].to_numpy()
                        weight = result.loc[j, 'weight_1':'weight_' + str(i)]
                    elif origin_inclusion == 'origin_dp':
                        X = result.loc[j, 'fit_dose_1':'fit_dose_' + str(i+1)].to_numpy()
                        y = result.loc[j, 'fit_response_1':'fit_response_' + str(i+1)].to_numpy()
                        weight = result.loc[j, 'weight_1':'weight_' + str(i+1)]
                    
                    if deg == 1:
                        if origin_inclusion == 'origin_int':
                            poly_reg = PolynomialFeatures(degree=deg)
                            X = X.reshape(-1,1)
                            X = poly_reg.fit_transform(X)
                            fitted_model = LinearRegression(fit_intercept=True).fit(X, y, weight)
                            result.loc[j, 'coeff_0x'] = fitted_model.coef_[0]
                            result.loc[j, 'coeff_1x'] = fitted_model.coef_[1]
                        else:
                            poly_reg = PolynomialFeatures(degree=deg)
                            X = X.reshape(-1,1)
                            X = poly_reg.fit_transform(X)
                            fitted_model = LinearRegression(fit_intercept=False).fit(X, y, weight)
                            result.loc[j, 'coeff_0x'] = fitted_model.coef_[0]
                            result.loc[j, 'coeff_1x'] = fitted_model.coef_[1]
                    else:
                        if origin_inclusion == 'origin_int':
                            poly_reg = PolynomialFeatures(degree=deg)
                            X = X.reshape(-1,1)
                            X = poly_reg.fit_transform(X)
                            fitted_model = LinearRegression(fit_intercept=True).fit(X, y, weight)
                            result.loc[j, 'coeff_0x'] = fitted_model.coef_[0]
                            result.loc[j, 'coeff_1x'] = fitted_model.coef_[1]
                            result.loc[j, 'coeff_2x'] = fitted_model.coef_[2]
                        else:
                            poly_reg = PolynomialFeatures(degree=deg)
                            X = X.reshape(-1,1)
                            X = poly_reg.fit_transform(X)
                            fitted_model = LinearRegression(fit_intercept=False).fit(X, y, weight)
                            result.loc[j, 'coeff_0x'] = fitted_model.coef_[0]
                            result.loc[j, 'coeff_1x'] = fitted_model.coef_[1]
                            result.loc[j, 'coeff_2x'] = fitted_model.coef_[2]
                          
                    # Calculate prediction and deviation
                    prediction = fitted_model.predict(poly_reg.fit_transform([[cal_pred.loc[i, 'dose']]]))[0]

                    result.loc[j, 'prediction'] = prediction
                    deviation = cal_pred.loc[i, 'response'] - prediction
                    result.loc[j, 'deviation'] = deviation
                    abs_deviation = abs(deviation)
                    result.loc[j, 'abs_deviation'] = abs_deviation

                # Second prediction onwards
                else:               
                    result.loc[j, 'patient'] = cal_pred.loc[i, 'patient']
                    result.loc[j, 'method'] = method_string
                    result.loc[j, 'pred_day'] = cal_pred.loc[i, 'day']
                    result.loc[j, 'dose'] = cal_pred.loc[i, 'dose']
                    result.loc[j, 'response'] = cal_pred.loc[i, 'response']

                    # Fill in previous coeff and deviation
                    result.loc[j, 'prev_coeff_2x'] = result.loc[j-1, 'coeff_2x']
                    result.loc[j, 'prev_coeff_1x'] = result.loc[j-1, 'coeff_1x']
                    result.loc[j, 'prev_coeff_0x'] = result.loc[j-1, 'coeff_0x']
                    result.loc[j, 'prev_deviation'] = result.loc[j-1, 'deviation']

                    # Shift last coefficient by previous deviation
                    result.loc[j, 'coeff_2x'] = result.loc[j, 'prev_coeff_2x']
                    result.loc[j, 'coeff_1x'] = result.loc[j, 'prev_coeff_1x']
                    result.loc[j, 'coeff_0x'] = result.loc[j, 'prev_coeff_0x'] + result.loc[j, 'prev_deviation']

                    # Calculate prediction and deviation
                    if deg == 1:
                        prediction = result.loc[j, 'dose'] * result.loc[j, 'coeff_1x'] +  result.loc[j, 'coeff_0x']
                    else:
                        prediction = (result.loc[j, 'dose'] ** 2) * result.loc[j, 'coeff_2x'] + result.loc[j, 'dose'] * result.loc[j, 'coeff_1x'] +  result.loc[j, 'coeff_0x']
                    
                    result.loc[j, 'prediction'] = prediction
                    deviation =  result.loc[j, 'response'] - prediction
                    result.loc[j, 'deviation'] = deviation
                    abs_deviation = abs(deviation)
                    result.loc[j, 'abs_deviation'] = abs_deviation

                j = j + 1

            list_of_result_df.append(result)

    else:
        
        result = result[0:0]

        j = 0

        for i in range(deg + 1, len(cal_pred)):

            # First prediction
            if j == 0:                    
                result.loc[j, 'patient'] = cal_pred.loc[i, 'patient']
                result.loc[j, 'method'] = method_string
                result.loc[j, 'pred_day'] = cal_pred.loc[i, 'day']

                if (origin_inclusion == 'wo_origin') or (origin_inclusion == 'origin_int'):
                    result.loc[j, 'fit_dose_1':'fit_dose_' + str(i)] = cal_pred.loc[0:i-1, 'dose'].to_numpy()
                    result.loc[j, 'fit_response_1':'fit_response_' + str(i)] = cal_pred.loc[0:i-1, 'response'].to_numpy()
                    result.loc[j, 'day_1':'day_' + str(i)] = cal_pred.loc[0:i-1, 'day'].to_numpy()
                elif origin_inclusion == 'origin_dp':
                    result.loc[j, 'fit_dose_1'] = 0
                    result.loc[j, 'fit_dose_2':'fit_dose_' + str(i+1)] = cal_pred.loc[0:i-1, 'dose'].to_numpy()
                    result.loc[j, 'fit_response_1'] = 0
                    result.loc[j, 'fit_response_2':'fit_response_' + str(i+1)] = cal_pred.loc[0:i-1, 'response'].to_numpy()
                    result.loc[j, 'day_1'] = cal_pred.loc[0, 'day'] - 1
                    result.loc[j, 'day_2':'day_' + str(i+1)] = cal_pred.loc[0:i-1, 'day'].to_numpy()

                result.loc[j, 'dose'] = cal_pred.loc[i, 'dose']
                result.loc[j, 'response'] = cal_pred.loc[i, 'response']

                # Curve fit equation
                if origin_inclusion == 'wo_origin' or origin_inclusion == 'origin_int':
                    X = result.loc[j, 'fit_dose_1':'fit_dose_' + str(deg+1)].to_numpy()
                    y = result.loc[j, 'fit_response_1':'fit_response_' + str(deg+1)].to_numpy()
                elif origin_inclusion == 'origin_dp':
                    X = result.loc[j, 'fit_dose_1':'fit_dose_' + str(deg+2)].to_numpy()
                    y = result.loc[j, 'fit_response_1':'fit_response_' + str(deg+2)].to_numpy()
                
                if deg == 1:
                    if origin_inclusion == 'origin_int':
                        poly_reg = PolynomialFeatures(degree=deg)
                        X = X.reshape(-1,1)
                        X = poly_reg.fit_transform(X)
                        fitted_model = LinearRegression(fit_intercept=True).fit(X, y)
                        result.loc[j, 'coeff_0x'] = fitted_model.coef_[0]
                        result.loc[j, 'coeff_1x'] = fitted_model.coef_[1]
                    else:
                        poly_reg = PolynomialFeatures(degree=deg)
                        X = X.reshape(-1,1)
                        X = poly_reg.fit_transform(X)
                        fitted_model = LinearRegression(fit_intercept=False).fit(X, y)
                        result.loc[j, 'coeff_0x'] = fitted_model.coef_[0]
                        result.loc[j, 'coeff_1x'] = fitted_model.coef_[1]
                else:
                    if origin_inclusion == 'origin_int':
                        poly_reg = PolynomialFeatures(degree=deg)
                        X = X.reshape(-1,1)
                        X = poly_reg.fit_transform(X)
                        fitted_model = LinearRegression(fit_intercept=True).fit(X, y)
                        result.loc[j, 'coeff_0x'] = fitted_model.coef_[0]
                        result.loc[j, 'coeff_1x'] = fitted_model.coef_[1]
                        result.loc[j, 'coeff_2x'] = fitted_model.coef_[2]
                    else:
                        poly_reg = PolynomialFeatures(degree=deg)
                        X = X.reshape(-1,1)
                        X = poly_reg.fit_transform(X)
                        fitted_model = LinearRegression(fit_intercept=False).fit(X, y)
                        result.loc[j, 'coeff_0x'] = fitted_model.coef_[0]
                        result.loc[j, 'coeff_1x'] = fitted_model.coef_[1]
                        result.loc[j, 'coeff_2x'] = fitted_model.coef_[2]
                    
                # Calculate prediction and deviation
                prediction = fitted_model.predict(poly_reg.fit_transform([[cal_pred.loc[i, 'dose']]]))[0]

                result.loc[j, 'prediction'] = prediction
                deviation = cal_pred.loc[i, 'response'] - prediction
                result.loc[j, 'deviation'] = deviation
                abs_deviation = abs(deviation)
                result.loc[j, 'abs_deviation'] = abs_deviation

            # Second prediction onwards
            else:               
                result.loc[j, 'patient'] = cal_pred.loc[i, 'patient']
                result.loc[j, 'method'] = method_string
                result.loc[j, 'pred_day'] = cal_pred.loc[i, 'day']
                result.loc[j, 'dose'] = cal_pred.loc[i, 'dose']
                result.loc[j, 'response'] = cal_pred.loc[i, 'response']

                # Fill in previous coeff and deviation
                result.loc[j, 'prev_coeff_2x'] = result.loc[j-1, 'coeff_2x']
                result.loc[j, 'prev_coeff_1x'] = result.loc[j-1, 'coeff_1x']
                result.loc[j, 'prev_coeff_0x'] = result.loc[j-1, 'coeff_0x']
                result.loc[j, 'prev_deviation'] = result.loc[j-1, 'deviation']

                # Shift last coefficient by previous deviation
                result.loc[j, 'coeff_2x'] = result.loc[j, 'prev_coeff_2x']
                result.loc[j, 'coeff_1x'] = result.loc[j, 'prev_coeff_1x']
                result.loc[j, 'coeff_0x'] = result.loc[j, 'prev_coeff_0x'] + result.loc[j, 'prev_deviation']

                # Calculate prediction and deviation
                if deg == 1:
                    prediction = result.loc[j, 'dose'] * result.loc[j, 'coeff_1x'] +  result.loc[j, 'coeff_0x']
                else:
                    prediction = (result.loc[j, 'dose'] ** 2) * result.loc[j, 'coeff_2x'] + result.loc[j, 'dose'] * result.loc[j, 'coeff_1x'] +  result.loc[j, 'coeff_0x']
                
                result.loc[j, 'prediction'] = prediction
                deviation =  result.loc[j, 'response'] - prediction
                result.loc[j, 'deviation'] = deviation
                abs_deviation = abs(deviation)
                result.loc[j, 'abs_deviation'] = abs_deviation

            j = j + 1

        list_of_result_df.append(result)
        
    return list_of_result_df

def RW(deg, cal_pred, result, method_string, list_of_result_df, origin_inclusion='wo_origin', tau=""):
    """
    Choose last deg + 1 unique dose-response pairs for RW as input for RW method,
    prepare input dataframe, apply method.

    Input:
    deg - degree of polynomial fit, 1 for linear, 2 for quadratic
    cal_pred - dataframe of calibration and efficacy-driven dosing data, 
                cal_pred_linear for linear, cal_pred_quad for quad
    result - dataframe of results
    method_string - string of method
    list_of_result_df - list of result dataframe from each patient

    Output:
    list_of_result_df
    """
    values = []
    indices = []
    
    j = 0

    result = result[0:0]

    half_life = np.arange(3.5, 41.5, 1)

    if tau == 1:

        # Loop through all half_lives
        for k in range(len(half_life)):

            result = result[0:0]

            j = 0

            # Loop through all rows and fill in result if enough unique dose-response pairs
            for i in range(0, len(cal_pred)-1):

                # Define new dose
                new_dose = cal_pred.loc[i, 'dose']

                # Add values and indices of first three unique doses into list
                if len(values) < deg + 1:

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
                counter_for_origin_int = 1
                if len(indices) == deg + 1:            

                    result.loc[j, 'patient'] = cal_pred.loc[i + 1, 'patient']
                    result.loc[j, 'method'] = method_string
                    result.loc[j, 'pred_day'] = cal_pred.loc[i + 1, 'day']
                    result.loc[j, 'half_life'] = half_life[k]
                    
                    # Fill in dose, response, day to fit model
                    cal_pred_string = ['dose', 'response', 'day']
                    result_string = ['fit_dose', 'fit_response', 'day']
                    for string_num in range(0,len(cal_pred_string)):
                        if (origin_inclusion == 'wo_origin') or (origin_inclusion == 'origin_int'):
                            result.loc[j, result_string[string_num] + '_1'] = cal_pred.loc[indices[0], cal_pred_string[string_num]]
                            result.loc[j, result_string[string_num] + '_2'] = cal_pred.loc[indices[1], cal_pred_string[string_num]]
                            if deg == 2:
                                result.loc[j, result_string[string_num] + '_3'] = cal_pred.loc[indices[2], cal_pred_string[string_num]]
                        elif origin_inclusion == 'origin_dp':
                            result.loc[j, result_string[string_num] + '_1'] = 0
                            result.loc[j, 'day_1'] = cal_pred.loc[indices[0], 'day'] - 1
                            result.loc[j, result_string[string_num] + '_2'] = cal_pred.loc[indices[0], cal_pred_string[string_num]]
                            result.loc[j, result_string[string_num] + '_3'] = cal_pred.loc[indices[1], cal_pred_string[string_num]]
                            if deg == 2:
                                result.loc[j, result_string[string_num] + '_4'] = cal_pred.loc[indices[2], cal_pred_string[string_num]]
                    
                    # Fill in new dose and response of prediction day
                    result.loc[j, 'dose'] = cal_pred.loc[i + 1, 'dose']
                    result.loc[j, 'response'] = cal_pred.loc[i + 1, 'response']

                    # Calculate and fill in weight
                    fixed_pred_day = result.loc[j, 'pred_day']
                    fixed_half_life = result.loc[j, 'half_life']
                    if (origin_inclusion == 'wo_origin') or (origin_inclusion =='origin_int'):
                        day_array = result.loc[j, 'day_1':'day_' + str(deg+1)].astype(float)

                        # Fill in weight array
                        result.loc[j, 'weight_1':'weight_' + str(deg+1)] = np.exp(-(24*(fixed_pred_day - day_array)*(math.log(2)/fixed_half_life))).to_numpy()

                    elif origin_inclusion == 'origin_dp':
                        day_array = result.loc[j, 'day_1':'day_' + str(deg+2)].astype(float)

                        # Fill in weight array
                        result.loc[j, 'weight_1':'weight_' + str(deg+2)] = np.exp(-(24*(fixed_pred_day - day_array)*(math.log(2)/fixed_half_life))).to_numpy()

                    # Curve fit equation
                    if origin_inclusion == 'wo_origin' or origin_inclusion == 'origin_int':
                        X = result.loc[j, 'fit_dose_1':'fit_dose_' + str(deg+1)].to_numpy()
                        y = result.loc[j, 'fit_response_1':'fit_response_' + str(deg+1)].to_numpy()
                        weight = result.loc[j, 'weight_1':'weight_' + str(deg+1)]
                    elif origin_inclusion == 'origin_dp':
                        X = result.loc[j, 'fit_dose_1':'fit_dose_' + str(deg+2)].to_numpy()
                        y = result.loc[j, 'fit_response_1':'fit_response_' + str(deg+2)].to_numpy()
                        weight = result.loc[j, 'weight_1':'weight_' + str(deg+2)]
                    
                    if deg == 1:
                        if origin_inclusion == 'origin_int':
                            poly_reg = PolynomialFeatures(degree=deg)
                            X = X.reshape(-1,1)
                            X = poly_reg.fit_transform(X)
                            fitted_model = LinearRegression(fit_intercept=True).fit(X, y, weight)
                            result.loc[j, 'coeff_0x'] = fitted_model.coef_[0]
                            result.loc[j, 'coeff_1x'] = fitted_model.coef_[1]
                        else:
                            poly_reg = PolynomialFeatures(degree=deg)
                            X = X.reshape(-1,1)
                            X = poly_reg.fit_transform(X)
                            fitted_model = LinearRegression(fit_intercept=False).fit(X, y, weight)
                            result.loc[j, 'coeff_0x'] = fitted_model.coef_[0]
                            result.loc[j, 'coeff_1x'] = fitted_model.coef_[1]
                    else:
                        if origin_inclusion == 'origin_int':
                            poly_reg = PolynomialFeatures(degree=deg)
                            X = X.reshape(-1,1)
                            X = poly_reg.fit_transform(X)
                            fitted_model = LinearRegression(fit_intercept=True).fit(X, y, weight)
                            result.loc[j, 'coeff_0x'] = fitted_model.coef_[0]
                            result.loc[j, 'coeff_1x'] = fitted_model.coef_[1]
                            result.loc[j, 'coeff_2x'] = fitted_model.coef_[2]
                        else:
                            poly_reg = PolynomialFeatures(degree=deg)
                            X = X.reshape(-1,1)
                            X = poly_reg.fit_transform(X)
                            fitted_model = LinearRegression(fit_intercept=False).fit(X, y, weight)
                            result.loc[j, 'coeff_0x'] = fitted_model.coef_[0]
                            result.loc[j, 'coeff_1x'] = fitted_model.coef_[1]
                            result.loc[j, 'coeff_2x'] = fitted_model.coef_[2]
                    
                    # Calculate prediction and deviation
                    prediction = fitted_model.predict(poly_reg.fit_transform([[cal_pred.loc[i, 'dose']]]))[0]
                        
                    result.loc[j, 'prediction'] = prediction
                    deviation = cal_pred.loc[i + 1, 'response'] - prediction
                    result.loc[j, 'deviation'] = deviation
                    abs_deviation = abs(deviation)
                    result.loc[j, 'abs_deviation'] = abs_deviation

                    counter_for_origin_int = counter_for_origin_int + 1
                    
                j = j + 1
            
            list_of_result_df.append(result)
        
    else:
        # Loop through rows
        for i in range(0, len(cal_pred)-1):

            # Define new dose
            new_dose = cal_pred.loc[i, 'dose']

            # Add values and indices of first three unique doses into list
            if len(values) < deg + 1:

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
            counter_for_origin_int = 1
            if len(indices) == deg + 1:            

                result.loc[j, 'patient'] = cal_pred.loc[i + 1, 'patient']
                result.loc[j, 'method'] = method_string
                result.loc[j, 'pred_day'] = cal_pred.loc[i + 1, 'day']
                
                # Fill in dose, response, day to fit model
                cal_pred_string = ['dose', 'response', 'day']
                result_string = ['fit_dose', 'fit_response', 'day']
                for string_num in range(0,len(cal_pred_string)):
                    if (origin_inclusion == 'wo_origin') or (origin_inclusion == 'origin_int'):
                        result.loc[j, result_string[string_num] + '_1'] = cal_pred.loc[indices[0], cal_pred_string[string_num]]
                        result.loc[j, result_string[string_num] + '_2'] = cal_pred.loc[indices[1], cal_pred_string[string_num]]
                        if deg == 2:
                            result.loc[j, result_string[string_num] + '_3'] = cal_pred.loc[indices[2], cal_pred_string[string_num]]
                    elif origin_inclusion == 'origin_dp':
                        result.loc[j, result_string[string_num] + '_1'] = 0
                        result.loc[j, 'day_1'] = cal_pred.loc[indices[0], 'day'] - 1
                        result.loc[j, result_string[string_num] + '_2'] = cal_pred.loc[indices[0], cal_pred_string[string_num]]
                        result.loc[j, result_string[string_num] + '_3'] = cal_pred.loc[indices[1], cal_pred_string[string_num]]
                        if deg == 2:
                            result.loc[j, result_string[string_num] + '_4'] = cal_pred.loc[indices[2], cal_pred_string[string_num]]
                
                # Fill in new dose and response of prediction day
                result.loc[j, 'dose'] = cal_pred.loc[i + 1, 'dose']
                result.loc[j, 'response'] = cal_pred.loc[i + 1, 'response']
                
                # Curve fit equation
                if origin_inclusion == 'wo_origin' or origin_inclusion == 'origin_int':
                    X = result.loc[j, 'fit_dose_1':'fit_dose_' + str(deg+1)].to_numpy()
                    y = result.loc[j, 'fit_response_1':'fit_response_' + str(deg+1)].to_numpy()
                elif origin_inclusion == 'origin_dp':
                    X = result.loc[j, 'fit_dose_1':'fit_dose_' + str(deg+2)].to_numpy()
                    y = result.loc[j, 'fit_response_1':'fit_response_' + str(deg+2)].to_numpy()
                
                if deg == 1:
                    if origin_inclusion == 'origin_int':
                        poly_reg = PolynomialFeatures(degree=deg)
                        X = X.reshape(-1,1)
                        X = poly_reg.fit_transform(X)
                        fitted_model = LinearRegression(fit_intercept=True).fit(X, y)
                        result.loc[j, 'coeff_0x'] = fitted_model.coef_[0]
                        result.loc[j, 'coeff_1x'] = fitted_model.coef_[1]
                    else:
                        poly_reg = PolynomialFeatures(degree=deg)
                        X = X.reshape(-1,1)
                        X = poly_reg.fit_transform(X)
                        fitted_model = LinearRegression(fit_intercept=False).fit(X, y)
                        result.loc[j, 'coeff_0x'] = fitted_model.coef_[0]
                        result.loc[j, 'coeff_1x'] = fitted_model.coef_[1]
                else:
                    if origin_inclusion == 'origin_int':
                        poly_reg = PolynomialFeatures(degree=deg)
                        X = X.reshape(-1,1)
                        X = poly_reg.fit_transform(X)
                        fitted_model = LinearRegression(fit_intercept=True).fit(X, y)
                        result.loc[j, 'coeff_0x'] = fitted_model.coef_[0]
                        result.loc[j, 'coeff_1x'] = fitted_model.coef_[1]
                        result.loc[j, 'coeff_2x'] = fitted_model.coef_[2]
                    else:
                        poly_reg = PolynomialFeatures(degree=deg)
                        X = X.reshape(-1,1)
                        X = poly_reg.fit_transform(X)
                        fitted_model = LinearRegression(fit_intercept=False).fit(X, y)
                        result.loc[j, 'coeff_0x'] = fitted_model.coef_[0]
                        result.loc[j, 'coeff_1x'] = fitted_model.coef_[1]
                        result.loc[j, 'coeff_2x'] = fitted_model.coef_[2]
                    
                # Calculate prediction and deviation
                prediction = fitted_model.predict(poly_reg.fit_transform([[cal_pred.loc[i, 'dose']]]))[0]

                result.loc[j, 'prediction'] = prediction
                deviation = cal_pred.loc[i + 1, 'response'] - prediction
                result.loc[j, 'deviation'] = deviation
                abs_deviation = abs(deviation)
                result.loc[j, 'abs_deviation'] = abs_deviation

                counter_for_origin_int = counter_for_origin_int + 1
                
            j = j + 1
        
        list_of_result_df.append(result)

    return list_of_result_df

# Combine dataframes of individual patients
def format_result_df(cal_pred, result_df):
    """
    Rerrange column names, make patient and prediction day series of integers, sort, reset index
    
    Input: 
    cal_pred - dataframe of calibration and efficacy-driven dosing data for all patients
    result_df - results after applying all methods to patients
    
    Output:
    result_df - Formatted result_df
    """
    max_count_input = cal_pred[cal_pred['type']=='linear'].groupby('patient').count().max()['dose']
    col_names = ['patient', 'method', 'pred_day'] + \
                ['fit_dose_' + str(i) for i in range(1, max_count_input + 1)] + \
                ['fit_response_' + str(i) for i in range(1, max_count_input + 1)] + \
                ['day_' + str(i) for i in range(1, max_count_input + 1)] + \
                ['weight_' + str(i) for i in range(1, max_count_input + 1)] + \
                ['half_life', 'dose', 'response', 'prev_coeff_2x', 'prev_coeff_1x', 'prev_coeff_0x',\
                 'prev_deviation', 'coeff_2x', 'coeff_1x', 'coeff_0x', 'prediction', 'deviation',
                 'abs_deviation']
    result_df = result_df[col_names]
    result_df.patient = result_df.patient.apply(int)
    result_df.pred_day = result_df.pred_day.apply(int)
    result_df.sort_values(['patient', 'method', 'half_life', 'pred_day'], inplace=True)
    result_df = result_df.reset_index(drop=True)
    
    return result_df