import os
import requests
from zipfile import ZipFile
import pandas as pd

from ._base import DomainAwareDataset, get_data_home


def download_dataset(url, dest_folder):
    # Check if the destination folder exists, if not, create it
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)

    # Extract the filename from the URL
    file_name = url.split("/")[-1]

    # Check if the file already exists
    if not os.path.exists(os.path.join(dest_folder, file_name)):
        # Download the zip file
        response = requests.get(url)
        with open(os.path.join(dest_folder, file_name), 'wb') as zip_file:
            zip_file.write(response.content)

        # Extract the contents of the zip file
        with ZipFile(os.path.join(dest_folder, file_name), 'r') as zip_ref:
            zip_ref.extractall(dest_folder)

        print(f"Dataset downloaded and extracted to {dest_folder}")
    else:
        print(f"Dataset already exists in {dest_folder}")

def read_dataset(data_folder):
    # Read CSV files into DataFrames
    ids_mapping_df = pd.read_csv(os.path.join(data_folder, 'IDS_mapping.csv'), header=None, index_col=False)
    diabetic_data_df = pd.read_csv(os.path.join(data_folder, 'diabetic_data.csv'))


    # Identify the indices where sections change
    section_indices = ids_mapping_df[ids_mapping_df.isnull().all(axis=1)].index
    section_indices = [-1] + section_indices.tolist() + [len(ids_mapping_df)]


    # Split the DataFrame into sections based on the identified indices
    sections = [ids_mapping_df.iloc[section_indices[i]+1:section_indices[i+1]] for i in range(len(section_indices)-1)]
    sections = [section.reset_index() for section in sections]
    sections = [section.drop(columns=['index']) for section in sections]
    sections = [section.rename(columns=section.iloc[0]).drop(section.index[0]) for section in sections]

    # Assign each section to its corresponding DataFrame
    admission_type_df, discharge_disposition_df, admission_source_df = sections

    return admission_type_df, discharge_disposition_df, admission_source_df, diabetic_data_df

def preprocess_dataset(admission_type_df, discharge_disposition_df, admission_source_df, diabetic_data_df):
    """ https://tableshift.org/datasets.html#diabetes
    For the Diabetes prediction task, we use a set of features related to several known indicators
    for diabetes derived from. These risk factors are general physical health, high cholesterol, 
    BMI/obesity, smoking, the presence of other chronic health conditions (stroke, coronary heart diseas), 
    diet, alcohol consumption, exercise, household income, marital status, time since last checkup, 
    education level, health care coverage, and mental health. 
    For each risk factor, we extract a set of relevant features from the BRFSS foxed core and rotating 
    core questionnaires. We also use a shared set of demographic indicators (race, sex, state, survey year, 
    and a question related to income level). The prediction target is a binary indicator for whether 
    the respondent has ever been told they have diabetes.
    """
    
    # We dont need these id columns
    diabetic_data_df = diabetic_data_df.drop(['encounter_id', 'patient_nbr'], axis=1)
                          
    # Not enough non-null values in these columns
    diabetic_data_df = diabetic_data_df.drop(['max_glu_serum', 'A1Cresult'], axis=1)

    # Converted to binary (readmit vs. no readmit).
    # The readmitted column is the target variable.
    diabetic_data_df.loc[:, 'readmitted'] = diabetic_data_df['readmitted'].apply(lambda x: 0 if x == 'NO' else 1)

    # Drop rows with 'Unknown/Invalid' value (only 3 rows)
    diabetic_data_df = diabetic_data_df.loc[diabetic_data_df['gender'] != 'Unknown/Invalid']

    # Convert 'gender' to binary
    diabetic_data_df.loc[:, 'gender'] = diabetic_data_df['gender'].map({'Female': 0, 'Male': 1})

    # Drop weight column (97% missing values)
    diabetic_data_df = diabetic_data_df.drop(['weight'], axis=1)

    # Drop medical_specialty column (49% missing values)
    diabetic_data_df = diabetic_data_df.drop(['medical_specialty'], axis=1)

    # Drop payer_code column (40% missing values)
    diabetic_data_df = diabetic_data_df.drop(['payer_code'], axis=1)

    # Drop columns diag_1, diag_2, diag_3 (too many categories)
    # + Some are integers, some are floats, some are strings
    # TODO: Clean all these to be integers
    diabetic_data_df = diabetic_data_df.drop(['diag_1', 'diag_2', 'diag_3'], axis=1)

    # Define a mapping for each age range to its midpoint
    age_mapping = {
        '[70-80)': 75,
        '[60-70)': 65,
        '[50-60)': 55,
        '[80-90)': 85,
        '[40-50)': 45,
        '[30-40)': 35,
        '[90-100)': 95,
        '[20-30)': 25,
        '[10-20)': 15,
        '[0-10)': 5
    }

    # Map the 'age' column using the defined mapping
    diabetic_data_df.loc[:, 'age'] = diabetic_data_df['age'].map(age_mapping)
    

    columns_to_binary = [
    'diabetesMed', 'change', 'metformin-pioglitazone', 'metformin-rosiglitazone',
    'glimepiride-pioglitazone', 'glipizide-metformin', 'metformin', 'repaglinide',
       'nateglinide', 'chlorpropamide', 'glimepiride', 'acetohexamide',
       'glipizide', 'glyburide', 'tolbutamide', 'pioglitazone',
       'rosiglitazone', 'acarbose', 'miglitol', 'troglitazone', 'tolazamide',
       'examide', 'citoglipton', 'insulin', 'glyburide-metformin']

    diabetic_data_df[columns_to_binary] = diabetic_data_df[columns_to_binary].apply(
        lambda x: x.map({'No': 0, 'Down': 0, 'Steady': 1, 'Up': 1, 'Yes': 1, "Ch": 1}))

    return admission_type_df, discharge_disposition_df, admission_source_df, diabetic_data_df 


def generate_domain_aware_dataset(diabetic_data_df):
    dataset = DomainAwareDataset()

    # Group the DataFrame by the 'race' column
    grouped_df = diabetic_data_df.groupby('race')

    # Create separate DataFrames for each race
    race_dfs = {race: group for race, group in grouped_df}

    for domain_name in diabetic_data_df['race'].unique():
        race_df = race_dfs[domain_name]

        race_df = race_df.drop(['race'], axis=1)

        X = race_df.iloc[:, race_df.columns != 'readmitted'].values
        y = race_df['readmitted'].values

        dataset.add_domain(X, y, domain_name=domain_name)

    return dataset


def fetch_diabetes_dataset(only_domain_aware=False):
    # URL of the dataset
    dataset_url = "https://archive.ics.uci.edu/static/public/296/diabetes+130-us+hospitals+for+years+1999-2008.zip"

    # Destination folder for the dataset
    data_home = get_data_home(None)
    destination_folder = os.path.join(data_home, "diabetes_dataset")

    # Download the dataset
    download_dataset(dataset_url, destination_folder)

    # Read the dataset into DataFrames
    admission_type_df, discharge_disposition_df, admission_source_df, diabetic_data_df = read_dataset(destination_folder)

    # Preprocess the dataset
    admission_type_df, discharge_disposition_df, admission_source_df, diabetic_data_df = preprocess_dataset(admission_type_df, discharge_disposition_df, admission_source_df, diabetic_data_df)

    # Generate the domain-aware dataset
    dataset = generate_domain_aware_dataset(diabetic_data_df)

    if only_domain_aware:
        return dataset
    
    return dataset, admission_type_df, discharge_disposition_df, admission_source_df, diabetic_data_df
