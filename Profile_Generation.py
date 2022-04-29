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

def keep_ideal_data(df):
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

    return df