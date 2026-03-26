import os
import pandas as pd
import numpy as np
import pyreadstat

def process_nhanes(raw_data_dir, output_dir):
    """
    Loads, merges, and cleans the NHANES data, creating a non-leaky feature set.
    """
    print("Processing NHANES data...")
    
    # --- MERGE CYCLES ---
    nhanes_cycles = {}
    cycle_dirs = {
        '2015-2016': ('DEMO_I.xpt', os.path.join(raw_data_dir, 'NHANES', '2015-2016')),
        '2017-2020': ('P_DEMO.xpt', os.path.join(raw_data_dir, 'NHANES', '2017-March 2020 Pre-Pandemic'))
    }

    for cycle_name, (demo_file, folder_path) in cycle_dirs.items():
        print(f"  - Merging cycle: {cycle_name}")
        xpt_files = [f for f in os.listdir(folder_path) if f.lower().endswith('.xpt')]
        base_df, meta = pyreadstat.read_xport(os.path.join(folder_path, demo_file), encoding='latin1')
        
        for filename in xpt_files:
            if filename.lower() != demo_file.lower():
                temp_df, meta = pyreadstat.read_xport(os.path.join(folder_path, filename), encoding='latin1')
                cols_to_drop = [col for col in temp_df.columns if col in base_df.columns and col != 'SEQN']
                temp_df = temp_df.drop(columns=cols_to_drop)
                base_df = pd.merge(base_df, temp_df, on='SEQN', how='left')
        
        nhanes_cycles[cycle_name] = base_df

    nhanes_full_df = pd.concat(nhanes_cycles.values(), ignore_index=True)

    # --- CREATE FINAL, NON-LEAKY DATASET ---
    # 1. Whitelist the columns we need (including raw diagnostic ones for now)
    columns_to_keep = [
        'SEQN', 'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'BMXBMI', 'BPXSY1', 'BPXDI1', 
        'SMQ020', 'PAQ650', 'MCQ160C', 'MCQ160F', 'DIQ010', 'DIQ050', 
        'LBXTC', 'LBDHDD', 'LBXGH', 'LBXGLU' # Keep raw diagnostics temporarily
    ]
    # Ensure all columns exist before selecting
    columns_to_keep = [col for col in columns_to_keep if col in nhanes_full_df.columns]
    nhanes_selected_df = nhanes_full_df[columns_to_keep].copy()

    # 2. Create the accurate outcome variable
    nhanes_selected_df['Diabetes_Outcome'] = np.where(
        (nhanes_selected_df['DIQ010'] == 1) | 
        (nhanes_selected_df['LBXGH'] >= 6.5) | 
        (nhanes_selected_df['LBXGLU'] >= 126), 
        1, 0
    )

    # 3. Create the final feature set by DROPPING leaky and ID columns
    final_feature_columns = [
        'RIDAGEYR', 'RIAGENDR', 'RIDRETH3', 'BMXBMI', 'BPXSY1', 'BPXDI1', 
        'SMQ020', 'PAQ650', 'MCQ160C', 'MCQ160F', 'DIQ050', 'LBXTC', 'LBDHDD'
    ]
    nhanes_final_df = nhanes_selected_df[final_feature_columns + ['Diabetes_Outcome']].copy()

    # 4. Rename the cleaned columns to conceptual names
    rename_dict = {
        'RIDAGEYR': 'Age', 'RIAGENDR': 'Gender', 'RIDRETH3': 'Race_Ethnicity',
        'BMXBMI': 'BMI', 'BPXSY1': 'Systolic_BP', 'BPXDI1': 'Diastolic_BP',
        'SMQ020': 'Ever_Smoked_100_Cigs', 'PAQ650': 'Vigorous_Activity',
        'MCQ160C': 'History_Heart_Attack', 'MCQ160F': 'History_Stroke',
        'DIQ050': 'Family_History_Diabetes', 'LBXTC': 'Total_Cholesterol',
        'LBDHDD': 'HDL_Cholesterol'
    }
    nhanes_final_df = nhanes_final_df.rename(columns=rename_dict)
    
    # Save the processed file
    output_path = os.path.join(output_dir, 'nhanes_final.csv')
    nhanes_final_df.to_csv(output_path, index=False)
    print("NHANES processing complete.")

def process_brfss(raw_data_dir, output_dir):
    """
    Loads, merges, cleans, and harmonizes the BRFSS data.
    """
    print("Processing BRFSS data...")
    brfss_dfs = []
    brfss_base_dir = os.path.join(raw_data_dir, 'BRFSS')
    for year in ['2020', '2021', '2022']:
        filename = f'LLCP{year}.XPT'
        path = os.path.join(brfss_base_dir, year, filename)
        df, meta = pyreadstat.read_xport(path, encoding='latin1')
        brfss_dfs.append(df)
    
    brfss_full_df = pd.concat(brfss_dfs, ignore_index=True)

    brfss_columns_to_keep = [
        'DIABETE4', '_AGEG5YR', 'SEXVAR', '_RACE', '_BMI5', 'GENHLTH', 
        '_SMOKER3', '_TOTINDA', 'CVDINFR4', 'CVDSTRK3',
    ]
    brfss_selected_df = brfss_full_df[brfss_columns_to_keep].copy()

    brfss_rename_dict = {
        'DIABETE4': 'Diabetes_Outcome', '_AGEG5YR': 'Age', 'SEXVAR': 'Gender', 
        '_RACE': 'Race_Ethnicity', '_BMI5': 'BMI', 'GENHLTH': 'General_Health', 
        '_SMOKER3': 'Smoking_Status', '_TOTINDA': 'Physical_Activity', 
        'CVDINFR4': 'History_Heart_Attack', 'CVDSTRK3': 'History_Stroke'
    }
    brfss_clean_df = brfss_selected_df.rename(columns=brfss_rename_dict)

    brfss_clean_df['BMI'] = brfss_clean_df['BMI'] / 100.0
    brfss_clean_df['Diabetes_Outcome'] = brfss_clean_df['Diabetes_Outcome'].replace({1: 1, 3: 0, 4: 0, 2: 0, 7: np.nan, 9: np.nan})
    age_map = { 1: 21.5, 2: 27.5, 3: 32.5, 4: 37.5, 5: 42.5, 6: 47.5, 7: 52.5, 8: 57.5, 9: 62.5, 10: 67.5, 11: 72.5, 12: 77.5, 13: 80.0, 14: np.nan }
    brfss_clean_df['Age'] = brfss_clean_df['Age'].map(age_map)
    brfss_clean_df['Gender'] = brfss_clean_df['Gender'].replace({1: 0, 2: 1}) # 0=Male, 1=Female
    brfss_clean_df['Smoking_Status'] = brfss_clean_df['Smoking_Status'].map({1: 1, 2: 1, 3: 1, 4: 0, 9: np.nan}) # 1=Ever, 0=Never
    
    # Corrected logic: Handle General_Health separately
    brfss_clean_df['General_Health'] = brfss_clean_df['General_Health'].replace({7: np.nan, 9: np.nan})
    
    # Harmonize the remaining simple Yes/No columns
    for col in ['Physical_Activity', 'History_Heart_Attack', 'History_Stroke']:
        brfss_clean_df[col] = brfss_clean_df[col].replace({1: 1, 2: 0, 7: np.nan, 9: np.nan})

    output_path = os.path.join(output_dir, 'brfss_final.csv')
    brfss_clean_df.to_csv(output_path, index=False)
    print("BRFSS processing complete.")

def process_pima(raw_data_dir, output_dir):
    """
    Loads, cleans, and harmonizes the PIMA dataset.
    """
    print("Processing PIMA data...")
    pima_path = os.path.join(raw_data_dir, 'PIMA', 'diabetes.csv')
    pima_df = pd.read_csv(pima_path)
    
    pima_rename_dict = {
        'Pregnancies': 'Pregnancies', 'Glucose': 'Fasting_Glucose',
        'BloodPressure': 'Diastolic_BP', 'SkinThickness': 'Skin_Thickness',
        'Insulin': 'Insulin', 'BMI': 'BMI',
        'DiabetesPedigreeFunction': 'Family_History_Diabetes', 'Age': 'Age',
        'Outcome': 'Diabetes_Outcome'
    }
    pima_clean_df = pima_df.rename(columns=pima_rename_dict)
    
    output_path = os.path.join(output_dir, 'pima_final.csv')
    pima_clean_df.to_csv(output_path, index=False)
    print("PIMA processing complete.")

def main():
    """
    Main function to execute the data processing pipeline.
    """
    project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    raw_data_dir = os.path.join(project_dir, 'data', '01_raw')
    processed_data_dir = os.path.join(project_dir, 'data', '03_processed')

    os.makedirs(processed_data_dir, exist_ok=True)
    
    process_nhanes(raw_data_dir, processed_data_dir)
    process_brfss(raw_data_dir, processed_data_dir)
    process_pima(raw_data_dir, processed_data_dir)
    
    print("\nData processing pipeline finished!")

if __name__ == '__main__':
    main()
    