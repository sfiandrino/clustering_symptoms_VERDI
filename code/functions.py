import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

def filter_dataset(df):
    """
    This function filters the dataset by keeping only children and adolescents under 20 years old, 
    excluding individuals without a baseline date (infection onset) and excluding individuals with negative test.
    Inputs:
        df: a pandas dataframe
    Outputs:
        df_f: a pandas dataframe filtered
    """

    #First Filter: keeping children and adolescent under 20 years old
    df_ff = df[df['age_enrollment'] < 20]
    print("Individuals aged more than 20 years old: ", df.shape[0] - df_ff.shape[0])
    #Second filter:  excluding individuals with negative test
    df_sf = df_ff[df_ff['baseline_date_infection'].notna()]
    #df_sf = df_ff[df_ff['NPH_swab_result'] == 1]
    #print("Individuals with negative test: ", (df_ff.shape[0] - df_sf.shape[0]))
    #Third filter: excluding individuals without a baseline date of infection
    #num_nat = df_sf['baseline_date_infection'].isna().sum()
    #df_tf = df_sf.dropna(subset=['baseline_date_infection'])
    print("Individuals without baseline date of infection: ", (df_ff.shape[0] - df_sf.shape[0]))
    return df_sf


def get_only_severeriskfactors(df):
    """
    This function filters the dataset by keeping only children and adolescents with severe risk factors.
    Inputs:
        df: a pandas dataframe
    Outputs:
        df_rf: a pandas dataframe filtered
    """
    #List of risk factors associated with severe pediatric COVID-19:
    #Chronic pulmonary conditions (e.g., bronchopulmonary dysplasia and uncontrolled asthma)
    #Cardiovascular conditions, (i e.g., congenital heart disease)
    #Immunocompromising conditions (e.g., malignancy, primary immunodeficiency, and immunosuppression)
    #Neurologic conditions (e.g., epilepsy and select chromosomal/genetic conditions)
    #Prematurity
    #Feeding tube dependence and other pre-existing technology dependence requirements
    #Diabetes mellitus
    #Obesity

    list_keep = ['com_asthma','com_obesity', 'com_prematurity',
                 'com_diabetes', 'com_chronic-resp-disease',
                 'com_congenital-heart-disease', 'com_neurological-disease',
                 'com_primary-immunodeficiency', 'com_immunosuppressant-treatment',
                 'com_tumor','com_TCSE','com_SOT']
    list_drop = ['com_malnutrition', 'com_tracheostomy',
                 'com_rheumatic-disease', 'com_neprhopathy',
                 'com_ematological-disease', 'com_HIV',
                 'com_chronic-hepatitis', 'com_metabolic-disorders',
                 'com_others', 'com_nocom']
    df_rf = df.drop(list_drop, axis = 1)
    return df_rf

def fix_covid_vaccination(df):
    """
    This function fix missing values by establishing if the individuals have received the COVID-19 vaccine.
    Inputs:
        df: a pandas dataframe
    Outputs:
        df_v: a revised pandas dataframe
    """
    # Approved COVID-19 vaccines for children and adolescents:
    # 1) vaccine for individuals aged 12 years and older --> approved on May 31, 2021 
    # 2) vaccine for individuals aged 5-11 years --> approved on December 1, 2021
    # 3) vaccine for individuals aged 6 months to 4 years --> not approved when the enrollment was open

    # Convert baseline_date_infection to datetime
    df['baseline_date_infection'] = pd.to_datetime(df['baseline_date_infection'], errors='coerce')

    # Get sub-dataframe of individuals with missing vaccination values
    df_n = df[df['vax_covid_firstdose'].isna()]
    print("Individuals with missing values:", df_n.shape[0])

    # Define vaccine approval dates
    date_12_plus = datetime(2021, 6, 1)
    date_5_to_11 = datetime(2021, 12, 1)

    # Fill missing values based on vaccine approval date
    index_1 = df_n[(df_n['baseline_date_infection'] < date_12_plus) & (df_n['age_enrollment'] >= 12)].index
    index_2 = df_n[(df_n['baseline_date_infection'] < date_5_to_11) & (df_n['age_enrollment'].between(5, 11))].index
    index_3 = df_n[df_n['age_enrollment'] < 5].index

    # Set missing vaccination status to 0 (not vaccinated)
    df.loc[index_1, 'vax_covid_firstdose'] = 0
    df.loc[index_2, 'vax_covid_firstdose'] = 0
    df.loc[index_3, 'vax_covid_firstdose'] = 0

    # if still some missing values, set them to -1
    df['vax_covid_firstdose'].fillna(-1, inplace=True)
    return df

def data_manipulation(df):
    # Change type of age column to integer
    df['age_enrollment'] = df['age_enrollment'].astype('int')
    # Replace row with 999 (outlier) with 5 (Caucasian) and fill nan values with 5
    df['ethnicity'] = df['ethnicity'].replace(999, 5)
    df['ethnicity'] = df['ethnicity'].fillna(5)
    df['ethnicity'] = df['ethnicity'].astype('int')
    # Replace nan values of columns referred to vaccination with unknown values
    df['vax_flu_2019-2020'] = df['vax_flu_2019-2020'].fillna(-1)
    df['vax_flu_2020_2021'] = df['vax_flu_2020_2021'].fillna(-1)
    # Change type of category for the gender column - 1=female; 0=male
    le = LabelEncoder()
    df['gender'] = le.fit_transform(df['gender'])  
    return df

def check_prevalence(df, cols_com):
    """
    This function checks the prevalence of each comorbidity in the dataset and removes the columns with no prevalence.
    Inputs:
        df: a pandas dataframe
        cols_com: a list of comorbidities
    Outputs:
        df_p: a pandas dataframe with the columns with a prevalence higher than 0
        list_removed: a list of removed columns
    """
    for col in cols_com:
        df[col] = df[col].astype(int)
    #Remove the column with a prevalence lower that 1%
    list_removed = []
    for col in cols_com:
        mask = df[col] == 1
        count = len(df[mask])
        N = count
        D = df.loc[:, col].shape[0]
        prevalence = N/D * 100
        print(col, prevalence)
        if prevalence == 0:
            list_removed.append(col)  
    df_p = df.drop(list_removed, axis = 1)
    return df_p, list_removed

def define_hex_vax(df):
    df['hex_vax'] = df[['vax_DTP','vax_IPV/OPV', 'vax_Hib', 'vax_HBV']].sum(axis=1)
    df['hex_vax'].replace([0, 1, 2, 3, 4], [0, 0, 0, 0, 1], inplace=True)
    return df


def extract_new_info(df):
    # Get total number of comorbidities
    cols_com = [col for col in df.columns if 'com_' in col]
    df['num_comorbidities'] = df[cols_com].sum(axis=1)
    # Create column at least one comorbidity
    df['at_least_one_comorbidity'] = 0
    df.loc[df['num_comorbidities'] > 0, 'at_least_one_comorbidity'] = 1

    # Get total number of symptoms (columns that start with 'sym_' but do not end with 'od' or 'td')
    cols_sym = [col for col in df.columns if 'sym_' in col and not col.endswith('od') and not col.endswith('td')]
    cols_sym.remove('sym_symptoms_other_text')
    df['num_symptoms'] = df[cols_sym].sum(axis=1)

    # Get duration of each symptom (columns that end with 'td' is the end of a symptom and 'od' is the onset of a symptom)
    cols_sym_onset = [col for col in df.columns if 'sym_' in col and col.endswith('od')]
    cols_sym_end = [col for col in df.columns if 'sym_' in col and col.endswith('td')]

    # Calcolare la durata per ciascun sintomo
    for i in range(len(cols_sym_onset)):
        onset = cols_sym_onset[i]
        end = cols_sym_end[i]
        # Estrai il nome del sintomo, rimuovendo la parte '_od' o '_td'
        sym = onset.split('_')[1]

         # Calcolare la durata come differenza in giorni, ignorando i NaT
        mask_valid_dates = df[onset].notna() & df[end].notna()  # Solo le righe con date valide
        df.loc[mask_valid_dates, f'duration_{sym}'] = (df.loc[mask_valid_dates, end] - df.loc[mask_valid_dates, onset]).dt.days

        # Gestire i casi in cui le date siano mancanti o invalidi, impostando la durata a NaN o 0
        df[f'duration_{sym}'] = df[f'duration_{sym}'].fillna(0).astype(int)  # Imposta a 0 se la durata è NaN
    
    # Get the median duration of all symtpoms (excluding 0 values to avoid bias)
    cols_duration = [col for col in df.columns if col.startswith('duration_')]
    df['median_sym_duration'] = df[cols_duration].replace(0, np.nan).median(axis=1, skipna=True)
    df['median_sym_duration'] = df['median_sym_duration'].fillna(0)

    # Make category_infection column based of median duration of infection
    # median duration of infection = 0: category 0
    # median duration of infection > 0 and <= 5: category 1
    # median duration of infection > 5: category 2
    df['infection_category'] = 0
    df.loc[df['median_sym_duration'] > 0, 'infection_category'] = 1
    df.loc[df['median_sym_duration'] >= 5, 'infection_category'] = 2

    # Add variant of infection (VOC) based on the date of infection
    d_preOmicron_start_str = datetime.strptime('01-02-2020', '%d-%m-%Y')
    d_preOmicron_end_str = datetime.strptime('14-12-2021', '%d-%m-%Y')
    d_Omicron_start_str = datetime.strptime('15-12-2021', '%d-%m-%Y')
    d_Omicron_end_str = datetime.strptime('08-12-2022', '%d-%m-%Y')

    df['baseline_date_infection'] = pd.to_datetime(df['baseline_date_infection'], errors='coerce')
    def get_variant(date_infection):
        if pd.isna(date_infection):
            return -1
        elif d_preOmicron_start_str <= date_infection <= d_preOmicron_end_str:
            return 0
        elif d_Omicron_start_str <= date_infection <= d_Omicron_end_str:
            return 1
        else:
            return -1
        
    df['VOC'] = df['baseline_date_infection'].apply(get_variant)
    df = define_hex_vax(df)

    return df

def remove_non_referable_symptoms(df_f_risk):
    df_f_risk = df_f_risk.drop(['sym_headache', 'sym_smell-taste-alterations', 
                                'sym_headache_od','sym_headache_td', 
                                'sym_smell-taste-alteration_od', 'sym_smell-taste-alteration_td'], axis=1)
    return df_f_risk
