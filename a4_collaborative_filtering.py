import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, classification_report
from Scratch.loggers import data_dir, data_out


def split_data_and_train(df_model):
    hike_attributes = [i for i in df_model.columns if i not in ['sentiment', 'reviewer_id',
                                                                'hike_id']]
    df_model.reset_index(inplace=True, drop=True)
    X = df_model[hike_attributes]
    y = df_model[['sentiment']]
    reviewer_ids = df_model[['reviewer_id']]
    hike_ids = df_model[['hike_id']]
    X_train, X_test, y_train, y_test, train_ids, test_ids, train_hike_ids, test_hike_ids = train_test_split(
        X, y, reviewer_ids, hike_ids, test_size=0.2, random_state=42, stratify=y)
    # so that I can keep track of everyone

    #fix index, inplace is outdated
    # X_train = X_train.reset_index(drop=True)
    # X_test = X_test.reset_index(drop=True)
    # y_train = y_train.reset_index(drop=True)
    # y_test = y_test.reset_index(drop=True)
    # train_ids = train_ids.reset_index(drop=True)
    # test_ids = test_ids.reset_index(drop=True)



    # model to get a baseline, and then make recommendations once accurate
    model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
    model_knn.fit(X_train)

    distances, indices = model_knn.kneighbors(X_test)
    #preds
    predicted_sentiments = []
    for idx in indices:
        neighbors_sentiments = y_train.iloc[idx]
        most_common_sentiment = neighbors_sentiments.mode().iloc[0]
        predicted_sentiments.append(most_common_sentiment)

    # Now see if the model is accurate
    accuracy = accuracy_score(y_test, predicted_sentiments)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, predicted_sentiments))
#     Accuracy: 0.9460327051625816
#               precision    recall  f1-score   support
#           -1       0.94      0.91      0.92      8826
#            0       0.96      0.99      0.98      8826
#            1       0.94      0.94      0.94      8827
#     accuracy                           0.95     26479
#    macro avg       0.95      0.95      0.95     26479
# weighted avg       0.95      0.95      0.95

    return X_train, X_test, y_train, y_test, train_ids, test_ids, model_knn, indices


## I want the ids in the test set to then get their highest possible recommendation from their
# nearest neighbors:
def get_recommendations(test_ids, indices, X_train, y_train, test_hike_ids):
    recommendations = []
    for reviewer_id in test_ids['reviewer_id'].unique():
        idx = test_ids[test_ids['reviewer_id'] == reviewer_id].index
        if not idx.empty:
            neighbor_idx = indices[idx[0]]
            similar_hikes = X_train.iloc[neighbor_idx]
            similar_hikes['sentiment'] = y_train.iloc[neighbor_idx].values
            similar_hikes['hike_id'] = test_hike_ids.iloc[neighbor_idx].values
            highest_sentiment_hike = similar_hikes.loc[similar_hikes['sentiment'].idxmax()]
            recommendations.append((reviewer_id, highest_sentiment_hike))

    # Check recommendations
    print('Recommendations for each unique reviewer ID in the test set:')
    for reviewer_id, hike in recommendations:
        print(f'Reviewer ID: {reviewer_id}')
        print(f'Recommended hike: {hike}')









if __name__ == '__main__':
    print('Start Filtering')

    df_model = pd.read_csv(data_out + 'model_data1_no_pca.csv')
    #df_model.set_index('hike_id', inplace=True)
    #df_final_pca.to_csv(data_out + 'model_data1_pca.csv', index=False)

    X_train, X_test, y_train, y_test, train_ids, test_ids, model_knn, indices = \
        split_data_and_train(df_model)

    # get_recommendations(test_ids, indices, X_train, y_train)

    #PCA
    print('NOW WITH PCA')
    df_model_pca = pd.read_csv(data_out + 'model_data1_pca.csv')

    X_train, X_test, y_train, y_test, train_ids, test_ids, model_knn, indices = \
        split_data_and_train(df_model)

    # get_recommendations(test_ids, indices, X_train, y_train)



#POTETNIAL FUTURE IMPROVEMENTS: Make a time factor,where it only observes hike recommendations from
# same time of year?
#94% accuracy, works again!