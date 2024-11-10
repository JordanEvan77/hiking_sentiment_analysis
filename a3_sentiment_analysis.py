from textblob import TextBlob
import pandas as pd
from Scratch.loggers import data_dir

def run_sentiment_check(text):
    '''This little helper function will iterate through each entry in the columns to
    return a rating'''
    blob = TextBlob(text)
    if blob.sentiment.polarity > 0:
        return  1 # Positive
    elif blob.sentiment.polarity == 0:
        return  0 # Mixed
    else:
        return  -1 # Negative

    #returns a range.



df_sent = pd.read_csv(data_dir + 'hiking_reports_36.csv')

df_sent['sentiment'] = df_sent['Report Text'].apply(run_sentiment_check)
