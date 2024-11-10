from textblob import TextBlob
import pandas as pd
from Scratch.loggers import data_dir

def run_sentiment_check(text):
    '''This little helper function will iterate through each entry in the columns to
    return a rating'''
    blob = TextBlob(str(text))
    if blob.sentiment.polarity > 0:
        return  1 # Positive
    elif blob.sentiment.polarity == 0:
        return  0 # Mixed
    else:
        return  -1 # Negative

    #returns a category, with KNN this is the best option for now, despite losing some of the nuance
    # of the finer floats. May adjust later



df_sent = pd.read_csv(data_dir + 'hiking_reports_36.csv')

df_sent['sentiment'] = df_sent['Report Text'].apply(run_sentiment_check)
# the sentiment takes about a second to do a couple hundred it looks like? so this step will take
# a couple minutes


# distribution:
df_sent['sentiment'].value_counts()
# I tested the first 5 different texts, and it looks like it does a good job
# but we definitely have a much larger amount of positive reviews than negative.