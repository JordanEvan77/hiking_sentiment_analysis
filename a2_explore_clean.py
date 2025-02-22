import ast
import sklearn
import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from Scratch.loggers import data_dir, data_out
from sklearn.preprocessing import LabelEncoder, StandardScaler
from hiking_sentiment_analysis.a3_sentiment_analysis import run_sentiment_check
plt.ion()


def initial_checks(df):
    '''
    A function for exploratory data analysis
    :param df:
    :return:
    '''
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
        }) # just for reference
        df_num = pd.concat([df_num, new_row], ignore_index=True)
    # null count isn't too severe, will try imputation and dropping outliers completely
    return df_num


def trim_text_clean_numeric(df):
    '''
    A funciton that handles a lot of manual string parsing
    :param df: Raw df uncleaned with messy strings
    :return: data frame with parsed columns
    '''
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
    return df



def create_cross_viz(df, cat_cols, id_cols, num_cols):
    '''
    Just a function for creating initial visualizations for EDA
    :param df: text parsed dataframe
    :param cat_cols: set of key categorical variables
    :param id_cols: set of key id columns
    :param num_cols: set of key num cols
    :return: visualizations
    '''
    df_viz = df.copy()
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


def fix_key_features(df):
    '''
    This problematic column needs to be encoded and parsed
    :param df: unparsed dataframe
    :return: data frame with new columns
    '''
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
    df_2 = df_2.drop('Key Features', axis=1)

    # 'Difficulty', Encoding:
    diff_map = {'Easy':0, 'Easy/Moderate':1, 'Moderate':2, 'Moderate/Hard':3, 'Hard':4}
    df_2['Difficulty'] = df_2['Difficulty'].map(diff_map) # this should be the only cleaning, beyond null cleaning
    # needed

    # 'Report Text',
    # this is the crucial part of the process where I get the signal, using sentiment analysis
    #  will use simple categories to start:
    # we don't want anything without a signal
    df_2 = df_2[~(df_2['Report Text'].isnull())]
    return df_2



def split_region(i):
    '''
    another simple string split function
    :param i:
    :return:
    '''
    if '>' in str(i):
        return str(i).split(' > ')
    else:
        return [str(i), None]


def split_condition(i):
    '''
    simple string split function
    :param i:
    :return:
    '''
    if ':' in str(i):
        return str(i).split(':')
    else:
        return [str(i), None]



def encode_cate_cols(df_2):
    '''
    Individually encodes specific categorical columns
    :param df_2:
    :return:
    '''
    reviewer_encoder = LabelEncoder()
    df_2['reviewer_id'] = reviewer_encoder.fit_transform(df_2['Trail Report By'])
    df_2['hike_id'] = reviewer_encoder.fit_transform(df_2['Hike Name'])
    df_2 = df_2.drop('Trail Report By', axis=1)
    df_2 = df_2.drop('Hike Name', axis=1)
    df_2.set_index(['hike_id', 'reviewer_id'], inplace=True)
    #TODO: Having index issues, track through process, and finalize a4 and a5! I want both to just be
    # columns!
    ##################################
    ########Date Features############
    ##################################
    df_2['Date'] = pd.to_datetime(df_2['Date'])
    df_2[['Day', 'Month', 'Year']] = df_2['Date'].apply(lambda x: pd.Series([int(x.day), int(x.month),
                                                                             int(x.year)]))
    df_2 = df_2.drop('Date', axis=1)
    return df_2


# ['Month', 'Day', 'Year' 'Rating', 'Highest Point', 'Elevation']
def drop_outliers(df, num_cols, threshold=1.5):
    '''
    A standardized helper function for dropping outliers
    :param df: dataframe with outliers
    :param num_cols: columns to check
    :param threshold: the iqr severity used
    :return: dataframe without outliers
    '''
    df_cleaned = df.copy()
    total_rows_lost = 0
    for col in num_cols:
        # IQR stuff
        Q1 = df_cleaned[col].quantile(0.25)
        Q3 = df_cleaned[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (threshold * IQR)
        upper_bound = Q3 + (threshold * IQR)

        rows_before = df_cleaned.shape[0]
        df_cleaned = df_cleaned[(df_cleaned[col] >= lower_bound) & (df_cleaned[col] <= upper_bound)]

        # get counts
        rows_after = df_cleaned.shape[0]
        rows_lost = rows_before - rows_after
        total_rows_lost += rows_lost

        print(f'{col} outliers removed, dropped {rows_lost} rows')
    print(f'Total rows lost: {total_rows_lost}')

    return df_cleaned


def impute_and_drop(df_3, num_cols):
    '''
    find count of missing values and impute with median
    :return:
    '''
    print(df_3[num_cols].isnull().sum())
    df_3[num_cols].fillna(df_3[num_cols].median(), inplace=True)

    # Check for duplicates
    print('dupes', df_3.duplicated().sum())

    # safe to drop?
    df_3.drop_duplicates(inplace=True)
    return df_3


def balance_classes(df_3):
    '''
    Handles class imbalance for model set up
    :param df_3: previously cleaned df
    :return: df_resampled: X variables that are aligned with rebalance and y_res which is aligned with rebalance
    '''
    smote = SMOTE(random_state=22) # I believe the minority class has a reasonable representation,
    # I just want more of them
    df_3.reset_index(inplace=True, drop=False)
    independent_vars = [i for i in df_3.columns if i != 'sentiment']
    df_3 = df_3.astype('float64')
    #df_3[independent_vars] = df_3[independent_vars].apply(pd.to_numeric)
    # bool_cols = df_3.select_dtypes(include='bool').columns
    # df_3[bool_cols] = df_3[bool_cols].astype('Int64')

    X = df_3[independent_vars]
    y = df_3[['sentiment']].astype('Int64')

    X_res, y_res = smote.fit_resample(X, y)
    df_resampled = pd.DataFrame(X_res, columns=independent_vars)
    #df_resampled['sentiment'] = y_res # wait
    df_resampled.set_index(['hike_id', 'reviewer_id'], inplace=True) # found index!
    y_res.index = df_resampled.index
    return df_resampled, y_res


def create_pca(df_standardized, y_res):
    '''
    Sets up PCA for alternate model test
    :param df_final:
    :return: PCA complete df
    '''
    df_final = df_standardized.copy()
    print('PCA')
    pca = PCA().fit(df_final)
    cumulative_explained_variance = np.cumsum(pca.explained_variance_ratio_)
    num_components = np.argmax(cumulative_explained_variance >= 0.90) + 1
    print('90% variance is captured at', num_components)

    plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.show()

    pca = PCA(n_components=num_components)
    df_final_pca = pd.DataFrame(pca.fit_transform(df_final))
    df_final_pca.index = df_final.index

    df_final['sentiment'] = y_res['sentiment']
    df_final_pca['sentiment'] = y_res['sentiment']

    #TODO: May want to do general feature selection over dimensionality reduction? for another option
    # in model?
    return df_final_pca, df_final





if __name__ == '__main__':
    print('Start')
    df_raw = pd.read_csv(data_dir + 'hiking_reports_64.csv')
    df = df_raw.copy()  # not too big to hold a copy in memory
    # don't need to do initial checks anymore, just move to trim
    df = trim_text_clean_numeric(df)


    ########### CROSS OVER VISUALS##############
    cat_cols = ['Key Features', 'Difficulty', 'Report Text', 'Region', 'Road', 'Bugs', 'Snow',
                'Type of Hike', 'Trail Conditions']
    id_cols = ['Hike Name', 'Trail Report By']
    num_cols = ['Date', 'Rating', 'Highest Point', 'Elevation']


    #create_cross_viz(df, cat_cols, id_cols, num_cols) # not used currently
    df_2 = fix_key_features(df)

    # do sentiment analysis encoding
    print('running sentiment')
    df_2['sentiment'] = df_2['Report Text'].apply(run_sentiment_check)


    # 'Region' fixes
    df_2['Region'].value_counts()
    # The region is made up of two parts, split it with string split, and maybe OHE the first
    #  portion? Then consider Label Encoding the second (too much variety with 60+?) and see later if
    #  the feature is important enough on the second region type (definitely keep the first,
    #  there should be less)
    df_2 = df_2[~(df_2['Region'].isnull())]  # due to region not being imputable
    df_2[['large_region', 'small_region']] = df_2['Region'].apply(split_region).apply(pd.Series)
    df_2['large_region'].value_counts()
    df_2['small_region'].value_counts()  # too many values for OHE and LE feels wrong, drop for now
    df_2 = df_2[[i for i in df_2.columns if i not in ['small_region', 'Region']]]


    #Parsing and Dummies~
    df_2[['general_trail_condition', 'detail_trail_condition']] = df_2['Trail Conditions'].apply(
        split_condition).apply(pd.Series)
    print(df_2['general_trail_condition'].value_counts())  # definitely OHE
    print(df_2['detail_trail_condition'].value_counts())  # could take each sub part of the text and
    # OHE it as well! But the general should be good for now, so OHE it and drop rest for now
    df_2 = df_2[[i for i in df_2.columns if i not in ['Report Text', 'Trail Conditions',
                                                      'detail_trail_condition']]]
    # 'Road',#'Bugs', 'Snow', 'Type of Hike' #  Typical OHE
    df_2 = pd.get_dummies(df_2, columns=['Road', 'Bugs', 'Snow', 'Type of Hike', 'large_region',
                                         'general_trail_condition'])

    #leaving this in to show my thought process:
    # Check what the count of nulls is now that encoding is done and we can potentially impute:
    cat_nulls = df_2[[i for i in df_2.columns if i not in id_cols + num_cols]].isna().sum()
    # Key Features           0
    # Report Text                                                           201
    # Difficulty                                                           1054

    # So difficulty is the one with the most nulls, but still less than 5% of the rows are null.
    # I could just drop the null values, but doing something like KNN to help fill the null values
    # may be worth a try. Especially since the elevation, max hieght and other features correlate
    # well with difficulty

    # The report text is our signal and has very few nulls, and the region is also a value that
    # contextually doesn't make a lot of sense to impute, and also has very few nulls.

    # impute after encoding!
    # 'Hike Name' as index??? sentiment analysis and this should run

    #necessary encoding
    df_2 = encode_cate_cols(df_2)


    # Impute Categorical
    impute = KNNImputer(n_neighbors=3)
    df_imputed = df_2.copy()
    imputed_values = impute.fit_transform(df_2[['Difficulty']])
    df_imputed['Difficulty'] = imputed_values


    ########### NUMERIC CLEANING ###########
    df_3 = df_imputed.copy()
    num_cols = ['Month', 'Day', 'Year', 'Rating', 'Highest Point', 'Elevation']


    #impute nuermic cols now, different approach and dupes
    df_3 = impute_and_drop(df_3, num_cols)


    # do outliers:
    df_3 = drop_outliers(df_3, num_cols)  # TODO: Do we maybe not want to drop month outliers?


    # Class Imbalance!
    df_resampled, y_res = balance_classes(df_3)


    # Scaling for KNN
    # Standardization OVER normalization for now
    print('scaling')
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(df_resampled)
    df_standardized = pd.DataFrame(standardized_data, columns=df_resampled.columns)
    df_standardized.index = df_resampled.index


    # Dimensionality reduction
    # I would like to try on two different datasets, one with and one without PCA. ALl other cleaning
    # steps are completely needed (outliers, nulls, feature engineering, encoding, class imbalance,
    # scaling)
    df_final_pca, df_final = create_pca(df_standardized, y_res)


    ########### DONE, Save out ###########
    # show alternate pipeline cleaning, as alternative
    df_final.reset_index(inplace=True, drop=False)
    df_final_pca.reset_index(inplace=True, drop=False)
    df_final.to_csv(data_out + 'model_data1_no_pca.csv')
    df_final_pca.to_csv(data_out + 'model_data1_pca.csv')
    print('Complete')