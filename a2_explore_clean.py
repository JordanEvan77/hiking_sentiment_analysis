import pandas as pd
import numpy as np
import sklearn
import missingno as msno
import matplotlib.pyplot as plt
import seaborn as sns
from Scratch.loggers import data_dir
import ast

plt.ion()

#Read in the data

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
# Note: Rating is on the hike overall, which is separate from trail reports, so each report doesn't
# have an attached rating, hence the need for sentiment analysis
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
########### CROSS OVER VISUALS##############
############################################
df_viz = df.copy()
cat_cols = ['Key Features', 'Difficulty', 'Report Text', 'Region', 'Road', 'Bugs', 'Snow',
            'Type of Hike', 'Trail Conditions']
id_cols = ['Hike Name', 'Trail Report By']
num_cols = ['Date', 'Rating', 'Highest Point', 'Elevation']


# scatter plot: Rating and Elevation
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_viz, x='Rating', y='Elevation')
plt.title('Scatter Plot: Rating and Elevation')
#plt.show() # gaps around 1 and 2 star ratings as anticipated, with highest hardest hikes

# scatter plot: Rating and Difficulty
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_viz, x='Rating', y='Difficulty')
plt.title('Scatter Plot: Rating and Difficulty')
#plt.show() # similar to above, nothing surprising

# scatter plot: Date and Difficulty
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df_viz, x='Date', y='Difficulty')
plt.title('Scatter Plot: Date and Difficulty')
#plt.show() # this should be done as count of per date

# line plot of count of difficult:
df_viz['Month'] = df_viz['Date'].dt.month
temp_viz = df_viz.groupby(['Month', 'Difficulty']).agg({'Difficulty':'count'}).rename(columns={
    'Difficulty':'Count'})
temp_viz.reset_index(inplace=True, drop=False)

plt.figure(figsize=(12, 6))
sns.lineplot(data=temp_viz, x='Month', y='Count', hue='Difficulty', marker='o')
plt.title('Count of Hikes per Difficulty Over Time')
plt.xticks(rotation=45)
plt.tight_layout()
#plt.show() # most difficult hiking is done in the summer, makes sense


#Bar Graph: Highest point avg per Difficulty
df_viz['Highest Point'] = pd.to_numeric(df_viz['Highest Point'], errors='coerce')
grouped_df = df_viz.groupby('Difficulty')['Highest Point'].mean().reset_index()
plt.figure(figsize=(10, 6))
sns.barplot(data=grouped_df, x='Difficulty', y='Highest Point')
plt.title('Bar Graph: Highest Point Avg per Difficulty')
#plt.show() # hard does have highest point

# Correlation heat map
corr = df_viz[num_cols].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap of Numerical Columns')
#plt.show() # rating and elevation, along with rate and highest point of course


############################################
########### CATEGORICAL CLEANING ###########
############################################
df_2 = df.copy() # next step of full cleaning

df_viz_g = df_viz.groupby('Key Features').size().reset_index(name='count')

plt.figure(figsize=(10, 6))
sns.barplot(data=df_viz_g, y='count', x='Key Features')
plt.title(f'bar of Key Features')
plt.xticks(rotation=45)
#plt.show()

df_viz_g = df_viz.groupby('Difficulty').size().reset_index(name='count')

plt.figure(figsize=(10, 6))
sns.barplot(data=df_viz_g, y='count', x='Difficulty')
plt.title(f'bar of Difficulty')
plt.xticks(rotation=45)
plt.show()

df_viz_g = df_viz.groupby('Report Text').size().reset_index(name='count')

plt.figure(figsize=(10, 6))
sns.barplot(data=df_viz_g, y='count', x='Report Text')
plt.title(f'bar of Report Text')
plt.xticks(rotation=45)
plt.show()

df_viz_g = df_viz.groupby('Region').size().reset_index(name='count')

plt.figure(figsize=(10, 6))
sns.barplot(data=df_viz_g, y='count', x='Region')
plt.title(f'bar of Region')
plt.xticks(rotation=45)
plt.show()

df_viz_g = df_viz.groupby('Road').size().reset_index(name='count')

plt.figure(figsize=(10, 6))
sns.barplot(data=df_viz_g, y ='count', x='Road')
plt.title(f'bar of Road')
plt.xticks(rotation=45)
plt.show()

df_viz_g = df_viz.groupby('Bugs').size().reset_index(name='count')

plt.figure(figsize=(10, 6))
sns.barplot(data=df_viz_g, y='count', x='Bugs')
plt.title(f'bar of Road')
plt.xticks(rotation=45)
plt.show()

df_viz_g = df_viz.groupby('Snow').size().reset_index(name='count')

plt.figure(figsize=(10, 6))
sns.barplot(data=df_viz_g, y='count', x='Snow')
plt.title(f'bar of Snow')
plt.xticks(rotation=45)
plt.show()

df_viz_g = df_viz.groupby('Type of Hike').size().reset_index(name='count')

plt.figure(figsize=(10, 6))
sns.barplot(data=df_viz_g, y='count', x='Type of Hike')
plt.title(f'bar of Type of Hike')
plt.xticks(rotation=45)
plt.show()

df_viz_g = df_viz.groupby('Trail Conditions').size().reset_index(name='count')

plt.figure(figsize=(10, 6))
sns.barplot(data=df_viz_g, y='count', x='Trail Conditions')
plt.title(f'bar of Trail Conditions')
plt.xticks(rotation=45)
plt.show()

#nbow check for Null values
plt.figure(figsize=(10, 6))
msno.matrix(df_viz[cat_cols])
plt.title('Missing Value Matrix')
plt.show()


#'Key Features',
# this one is one of the hardest to clean, I will need to parse it and hope to OHE:
key_feat_list = list(df_2['Key Features'].unique())
key_feat_list = [ast.literal_eval(list) for list in key_feat_list] # handle quoted lists
key_feat_list = [item for sublist in key_feat_list for item in sublist]

unique_items = list(set(key_feat_list)) # perfect, only 19 items
# then if the specific row entry has this in it, create 0 or 1
for new_col in unique_items:
    df_2[f'{new_col}_dummy'] = df_2['Key Features'].apply(lambda x: 1 if new_col in x else 0)
# this OHE type is preferred over LE, and will provide better performance


# 'Difficulty',


# 'Report Text',


# 'Region',


# 'Road',


#'Bugs',


# 'Snow',


# 'Type of Hike',


# 'Trail Conditions']


# TODO: Drop nulls, or do KNN imputation
df_final = []
df_knn = []



############################################
########### NUMERIC CLEANING ###########
############################################
df_3 = df_num



############################################
########### DONE, Save out ###########
############################################


#show alternate pipeline cleaning, as alternative


final_df = []
final_df.to_csv('data\model_ready\model_data1.csv', index=False)
