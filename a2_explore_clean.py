import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns

data_dir = 'data\\raw\\'

df_raw = pd.read_csv(data_dir + 'hiking_reports_23.csv')
df = df_raw.copy() # not too big to hold a copy in memory

#####################################
###########Initial Checks############
#####################################


for i in df.columns: print(df[i].head()) # 15 columns most categorical

cat_cols = ['Key Features', 'Difficulty', 'Report Text', 'Region', 'Road', 'Bugs', 'Snow',
            'Type of Hike', 'Trail Conditions']
id_cols = ['Hike Name', 'Trail Report By']
num_cols = ['Date', 'Rating', 'Highest Point', 'Elevation'] # date, float, int, int

temp = df.describe()

#check categorical:
df_cat= pd.DataFrame(columns=['Column', 'Num Categories', 'Category Counts'])

for col in cat_cols:
    num_categories = df[col].nunique()
    category_counts = df[col].value_counts().to_dict()
    null_count_pct = df[col].isnull().sum() / len(df) * 100
    new_row = pd.DataFrame({
        'Column': [col],
        'Num Categories': [num_categories],
        'Category Counts': [category_counts],
        'Null Count (%)': [null_count_pct]
    })
    df_cat = pd.concat([df_cat, new_row], ignore_index=True)
#Looks like a couple will be ranked an need LE 6: 'Difficulty', 'Road', 'Bugs', 'Snow',
    # 'Trail Conditions', 'Key Features' # too many categoricals, may simplify with sentiment?

# Some are unranked and will need OHE 2: 'Region', 'Type of Hike'

# Sentiment 1: 'Report Text'

# ID columns: 'Hike Name', 'Trail Report By', will be encoded as ID.

# Check Numeric

    #Trim text to numeric:
# int, remove feet and comma
df['Elevation'] = pd.to_numeric(df['Elevation'].str.replace(',', '').str.replace(' feet', ''),
                                errors='coerce').astype('Int64')
# int, remove feet and comma
df['Highest Point'] = pd.to_numeric(df['Highest Point'].str.replace(',', '').str.replace(
    ' feet', ''), errors='coerce').astype('Int64')
#float, remove ' out of 5'
df['Rating'] = pd.to_numeric(df['Rating'].str.replace(' out of 5', ''), errors='coerce').astype(
    'Float64')

#Date, remove '\n              '
df['Date'] = df['Date'].str.strip().str.replace('\n', '').str.strip()
df['Date'] = pd.to_datetime(df['Date'])


#now check nums
df_num = pd.DataFrame(columns=['Column', 'Null Count (%)', 'Outliers (%)'])

for col in num_cols:
    null_count_pct = df[col].isnull().sum() / len(df) * 100
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers_pct = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum() / len(df) * 100

    new_row = pd.DataFrame({
        'Column': [col],
        'Null Count (%)': [null_count_pct],
        'Outliers (%)': [outliers_pct]
    })
    df_num = pd.concat([df_num, new_row], ignore_index=True)
# null count isn't too severe, will try imputation and dropping outliers completely


############################################
########### CATEGORICAL CLEANING ###########
############################################
df2 = df_cat






############################################
########### NUMERIC CLEANING ###########
############################################
df_3 = df_num



############################################
########### DONE, Save out ###########
############################################


#show alternate pipeline cleaning, as alternative


final_df = []
final_df.to_csv('data\model\model_data1.csv', index=False)
