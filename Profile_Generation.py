from openpyxl import load_workbook
import pandas as pd

def get_sheet_names(input_file):
    """ Get sheet names which are also patient names """
    wb = load_workbook(input_file, read_only=True)
    patient_list = wb.sheetnames
    wb.close()
    return patient_list

def clean_data(patient_list, input_file, rows_to_skip):
    """
    Clean and append patient data from each sheet to dataframe.
    Read each sheet into dataframe, keep target columns, shift tac level one cell up,
    remove "mg"/"ng" from dose, add patient column
    
    Input:
    patient_list - sheet names which are also patient names
    input_file - raw data file
    rows_to_skip - rows to skip before dose-response data in raw data file
    
    Output:
    patient_df - filled dataframe with cleaned patient data    
    """
    # Create empty patient dataframe
    patient_df = pd.DataFrame(columns=["Day #", "Tac level (prior to am dose)", "Eff 24h Tac Dose"])

    # Loop through sheets
    for sheet in patient_list:
        # Read sheet into dataframe
        df = pd.read_excel(input_file, sheet_name=sheet, skiprows=rows_to_skip)

        # Keep target columns
        df = df[["Day #", "Tac level (prior to am dose)", "Eff 24h Tac Dose"]]

        # Shift tac level one cell up to match dose-response to one day
        df['Tac level (prior to am dose)'] = df['Tac level (prior to am dose)'].shift(-1)

        # Remove "mg"/"ng" from dose
        df['Eff 24h Tac Dose'] = df['Eff 24h Tac Dose'].astype(str).str.replace('mg', '')
        df['Eff 24h Tac Dose'] = df['Eff 24h Tac Dose'].astype(str).str.replace('ng', '')
        df['Eff 24h Tac Dose'] = df['Eff 24h Tac Dose'].astype(float)

        # Add patient column
        df['patient'] = sheet

        patient_df = patient_df.append(df)
        
    return patient_df
