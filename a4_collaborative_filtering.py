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
reviewer_ids = df_model[['reviewer_id']]

X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(X, y, reviewer_ids,
                           test_size=0.2, random_state=42)
# so that I can keep track of everyone

# model to get a baseline, and then make recommendations once accurate
model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(X)

distances, indices = model_knn.kneighbors(X_test)
predicted_sentiments = y_train.iloc[indices.flatten()].mode().iloc[0]

# Now see if the model is accurate
accuracy = accuracy_score(y_test, predicted_sentiments)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, predicted_sentiments))



## I want the ids in the test set to then get their highest possible recommendation from their
# nearest neighbors:
recommendations = []
for i, reviewer_id in enumerate(test_ids['reviewer_id'].unique()):
    idx = test_ids[test_ids['reviewer_id'] == reviewer_id].index
    if not idx.empty:
        neighbor_idx = indices[idx[0]]
        similar_hikes = X_train.iloc[neighbor_idx]
        similar_hikes['sentiment'] = y_train.iloc[neighbor_idx].values
        highest_sentiment_hike = similar_hikes.loc[similar_hikes['sentiment'].idxmax()]
        recommendations.append((reviewer_id, highest_sentiment_hike))

# Check recs
print('Recommendations for each unique reviewer ID in the test set:')
for reviewer_id, hike in recommendations:
    print(f'Reviewer ID: {reviewer_id}')
    print(f'Was reccomended hike: {hike}')









if __name__ == '__main__':
    print('Start Filtering')


#POTETNIAL FUTURE IMPROVEMENTS: Make a time factor,where it only observes hike recommendations from
# same time of year?