import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, classification_report

df_model = pd.read_csv('data\model_ready\model_data1.csv')


hike_attributes = [i for i in df_model.columns if i not in ['sentiment']]
X = df_model[hike_attributes]
y = df_model[['sentiment']]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# model to get a baseline, and then make recommendations once accurate
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(X)

distances, indices = model_knn.kneighbors(X_test)
predicted_sentiments = y_train.iloc[indices.flatten()].mode().iloc[0]

# Now see if the model is accurate
accuracy = accuracy_score(y_test, predicted_sentiments)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, predicted_sentiments))



## We want anything above .25, since I would say 0 is neutral/mixed, then .25 indicates enough
# enjoyment to recommend
positive_sentiment_hikes = df_model[df_model['sentiment']>= 0.25]
print('positive reviews to recommend', len(positive_sentiment_hikes))











# TODO: DO A MORE SPECIFIC FILTERING ALGO TO COMPARE


if __name__ == '__main__':
    print('Start Filtering')


#POTETNIAL FUTURE IMPROVEMENTS: Make a time factor,where it only observes hike recommendations from
# same time of year?