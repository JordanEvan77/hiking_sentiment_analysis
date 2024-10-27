from textblob import TextBlob
import pandas as pd

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



df_sent = pd.read_csv('data\model_ready\model_data1.csv')

df_sent['Sentiment'] = df_sent['Report Text'].apply(run_sentiment_check)
