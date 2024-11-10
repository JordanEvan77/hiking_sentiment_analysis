import pandas as pd
import numpy as np
import sklearn
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
from Scratch.loggers import data_dir
import ast

plt.ion()

#Read in the data

df_raw = pd.read_csv(data_dir + 'hiking_reports_36.csv')
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

viz_show = 'No'
if viz_show != 'No':
    # scatter plot: Rating and Elevation
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_viz, x='Rating', y='Elevation')
    plt.title('Scatter Plot: Rating and Elevation')
    plt.show() # gaps around 1 and 2 star ratings as anticipated, with highest hardest hikes

    # scatter plot: Rating and Difficulty
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_viz, x='Rating', y='Difficulty')
    plt.title('Scatter Plot: Rating and Difficulty')
    plt.show() # similar to above, nothing surprising

    # scatter plot: Date and Difficulty
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=df_viz, x='Date', y='Difficulty')
    plt.title('Scatter Plot: Date and Difficulty')
    plt.show() # this should be done as count of per date

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
    plt.show() # most difficult hiking is done in the summer, makes sense


    #Bar Graph: Highest point avg per Difficulty
    df_viz['Highest Point'] = pd.to_numeric(df_viz['Highest Point'], errors='coerce')
    grouped_df = df_viz.groupby('Difficulty')['Highest Point'].mean().reset_index()
    plt.figure(figsize=(10, 6))
    sns.barplot(data=grouped_df, x='Difficulty', y='Highest Point')
    plt.title('Bar Graph: Highest Point Avg per Difficulty')
    plt.show() # hard does have highest point

    # Correlation heat map
    corr = df_viz[num_cols].corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=.5)
    plt.title('Correlation Heatmap of Numerical Columns')
    plt.show() # rating and elevation, along with rate and highest point of course


############################################
########### CATEGORICAL CLEANING ###########
############################################

    df_viz_g = df_viz.groupby('Key Features').size().reset_index(name='count')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_viz_g, y='count', x='Key Features')
    plt.title(f'bar of Key Features')
    plt.xticks(rotation=45)
    plt.show() # clusters of common phrase sets, will need cleaning as expected

    df_viz_g = df_viz.groupby('Difficulty').size().reset_index(name='count')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_viz_g, y='count', x='Difficulty')
    plt.title(f'bar of Difficulty')
    plt.xticks(rotation=45)
    plt.show() # highest count is hard hikes, which shows desire for challenge vs rating?

    df_viz_g = df_viz.groupby('Report Text').size().reset_index(name='count')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_viz_g, y='count', x='Report Text')
    plt.title(f'bar of Report Text')
    plt.xticks(rotation=45)
    plt.show() # lots of unique values as expected, will need cleaning

    df_viz_g = df_viz.groupby('Region').size().reset_index(name='count')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_viz_g, y='count', x='Region')
    plt.title(f'bar of Region')
    plt.xticks(rotation=45)
    plt.show() # lots of reports in snoqualmie and cle elum

    df_viz_g = df_viz.groupby('Road').size().reset_index(name='count')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_viz_g, y ='count', x='Road')
    plt.title(f'bar of Road')
    plt.xticks(rotation=45)
    plt.show() # open for all vehicles is overwhelming majority, unexepected

    df_viz_g = df_viz.groupby('Bugs').size().reset_index(name='count')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_viz_g, y='count', x='Bugs')
    plt.title(f'bar of Road')
    plt.xticks(rotation=45)
    plt.show() # bugs are less of an  issue than anticipated, definitely has seasonal effect

    df_viz_g = df_viz.groupby('Snow').size().reset_index(name='count')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_viz_g, y='count', x='Snow')
    plt.title(f'bar of Snow')
    plt.xticks(rotation=45)
    plt.show() # snow free is dominant as expected

    df_viz_g = df_viz.groupby('Type of Hike').size().reset_index(name='count')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_viz_g, y='count', x='Type of Hike')
    plt.title(f'bar of Type of Hike')
    plt.xticks(rotation=45)
    plt.show() # a lot less over night and back packing reports than expected

    df_viz_g = df_viz.groupby('Trail Conditions').size().reset_index(name='count')

    plt.figure(figsize=(10, 6))
    sns.barplot(data=df_viz_g, y='count', x='Trail Conditions')
    plt.title(f'bar of Trail Conditions')
    plt.xticks(rotation=45)
    plt.show() # trail in good condition is overwhelming majority

    #nbow check for Null values
    plt.figure(figsize=(10, 6))
    msno.matrix(df_viz[cat_cols])
    plt.title('Missing Value Matrix')
    plt.show() # not a lot of paterns as to when the data is missing


#'Key Features',
# this one is one of the hardest to clean, I will need to parse it and hope to OHE:
df_2 = df.copy()
key_feat_list = list(df_2['Key Features'].unique())
key_feat_list = [ast.literal_eval(list) for list in key_feat_list] # handle quoted lists
key_feat_list = [item for sublist in key_feat_list for item in sublist]

unique_items = list(set(key_feat_list)) # perfect, only 19 items
# then if the specific row entry has this in it, create 0 or 1
for new_col in unique_items:
    df_2[f'{new_col}_dummy'] = df_2['Key Features'].apply(lambda x: 1 if new_col in x else 0)
# this OHE type is preferred over LE, and will provide better performance


# 'Difficulty', Encoding:
diff_map = {'Easy':0, 'Easy/Moderate':1, 'Moderate':2, 'Moderate/Hard':3, 'Hard':4}
df_2['Difficulty'] = df_2['Difficulty'].map(diff_map) # this should be the only cleaning, beyond null cleaning
# needed

# 'Report Text',
# this is the crucial part of the process where I get the signal, using sentiment analysis
#  will use simple categories to start:
from hiking_sentiment_analysis.a3_sentiment_analysis import run_sentiment_check
print('running sentiment')
df_2['sentiment'] = df_2['Report Text'].apply(run_sentiment_check)


# 'Region',
df_2['Region'].value_counts()
# The region is made up of two parts, split it with string split, and maybe OHE the first
#  portion? Then consider Label Encoding the second (too much variety with 60+?) and see later if
#  the feature is important enough on the second region type (definitely keep the first,
#  there should be less)
df_2 = df_2[~(df_2['Region'].isnull())] # due to region not being imputable
def split_region(i):
    if '>' in str(i):
        return str(i).split(' > ')
    else:
        return [str(i), None]

df_2[['large_region', 'small_region']] = df_2['Region'].apply(split_region).apply(pd.Series)
df_2['large_region'].value_counts()
df_2['small_region'].value_counts() # too many values for OHE and LE feels wrong, drop for now

df_2 = df_2[[i for i in df_2.columns if i not in ['small_region', 'Region']]]

# 'Trail Conditions']
df_2['Trail Conditions'].value_counts()
# This also has some segments to it. So for items with ':' grab the first portion as a column,
# and save the second as another column,
# should be able to OHE new first column, and maybe label encode rest? Depends on variety,
# may just dro pit?
def split_condition(i):
    if ':' in str(i):
        return str(i).split(':')
    else:
        return [str(i), None]

df_2[['general_trail_condition', 'detail_trail_condition']] = df_2['Trail Conditions'].apply(
    split_condition).apply(pd.Series)

print(df_2['general_trail_condition'].value_counts()) # definitely OHE
print(df_2['detail_trail_condition'].value_counts()) # could take each sub part of the text and
# OHE it as well! But the general should be good for now, so OHE it and drop rest for now
df_2 = df_2[[i for i in df_2.columns if i not in ['Trail Conditions', 'detail_trail_condition']]]

# 'Road',#'Bugs', 'Snow', 'Type of Hike',
#  Typical OHE
df_2 = pd.get_dummies(df_2, columns=['Road', 'Bugs', 'Snow', 'Type of Hike', 'large_region',
                                     'general_trail_condition', 'Key Features'])

# Check what the count of nulls is now that encoding is done and we can potentially impute:
cat_nulls = df_2[[i for i in df_2.columns if i not in id_cols+num_cols]].isna().sum()
# Key Features           0
# Report Text                                                           201
# Difficulty                                                           1054


#So difficulty is the one with the most nulls, but still less than 5% of the rows are null.
# I could just drop the null values, but doing something like KNN to help fill the null values
# may be worth a try. Especially since the elevation, max hieght and other features correlate
# well with difficulty

#The report text is our signal and has very few nulls, and the region is also a value that
# contextually doesn't make a lot of sense to impute, and also has very few nulls.

#drop nulls before I do imputing
df_2 = df_2[~(df_2['Report Text'].isnull())]


#impute after encoding!
#TODO: 'Hike Name' as index??? sentiment analysis and this should run
df.set_index('Name', inplace=True)
reviewer_encoder = LabelEncoder()
df_2['reviewer_id'] = reviewer_encoder.fit_transform(df_2['Report By'])


impute = KNNImputer(n_neighbors=3)
df_imputed = pd.DataFrame(impute.fit_transform(df_2), columns=['Difficulty'])



############################################
########### NUMERIC CLEANING ###########
############################################
df_3 = df_num





######################################
########Scaling for KNN###############
######################################
# I believe I want to scale even the binary variables for the model




##########################################
##########Dimensionality reduction########
##########################################
# I would like to try on two different datasets, one with and one without PCA


############################################
########### DONE, Save out ###########
############################################


#show alternate pipeline cleaning, as alternative


final_df = []
final_df_pca = []
final_df.to_csv('data\model_ready\model_data1_no_pca.csv', index=False)
final_df_pca.to_csv('data\model_ready\model_data1_no_pca.csv', index=False)

