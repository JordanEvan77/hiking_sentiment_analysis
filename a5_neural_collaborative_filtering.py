import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, classification_report
from Scratch.loggers import data_dir, data_out
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

def split_data_and_train(df_model):
    hike_attributes = [i for i in df_model.columns if i not in ['sentiment']]
    X = df_model[hike_attributes]
    y = df_model[['sentiment']]
    reviewer_ids = df_model[['reviewer_id']]

    X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(X, y, reviewer_ids,
                               test_size=0.2, random_state=42)
    # so that I can keep track of everyone

    #fix index, inplace is outdated
    X_train = X_train.reset_index(drop=True)
    X_test = X_test.reset_index(drop=True)
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)
    train_ids = train_ids.reset_index(drop=True)
    test_ids = test_ids.reset_index(drop=True)

    # model to get a baseline, and then make recommendations once accurate
    #---NCF Architecture:
    # Input Layers: User and item IDs.
    # Embedding Layers: Learn user and item representations.
    # Concatenation Layer: Combine user and item embeddings.
    # Dense Layers: Learn complex user-item interactions.
    # Output Layer: Provide predictions

    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')
    user_embedding = Embedding(input_dim=df_model['reviewer_id'].nunique(), output_dim=50,
                               name='user_embedding')(user_input)
    item_embedding = Embedding(input_dim=df_model['hike_id'].nunique(), output_dim=50,
                               name='item_embedding')(item_input)

    user_vector = Flatten(name='user_vector')(user_embedding)
    item_vector = Flatten(name='item_vector')(item_embedding)
    merged = Concatenate()([user_vector, item_vector])
    dense_1 = Dense(128, activation='relu')(merged)
    dense_2 = Dense(64, activation='relu')(dense_1)
    output = Dense(y_train.shape[1], activation='softmax')(dense_2)
    model_ncf = Model(inputs=[user_input, item_input], outputs=output)
    model_ncf.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy',
                      metrics=['accuracy'])
    model_ncf.fit([X_train['reviewer_id'], X_train['hike_id']], y_train, epochs=20, batch_size=32,
                  verbose=1)

    # get acc
    scores = model_ncf.evaluate([X_test['reviewer_id'], X_test['hike_id']], y_test, verbose=0)
    print(f"Accuracy: {scores[1]}")
    print(classification_report(np.argmax(y_test, axis=1),
                                np.argmax(model_ncf.predict([X_test['reviewer_id'],
                                                             X_test['hike_id']]), axis=1)))

    distances, indices = model_ncf.kneighbors(X_test)
    #preds
    predicted_sentiments = []
    for idx in indices: predicted_sentiments.append(y_train.iloc[idx].mode().iloc[0])

    # Now see if the model is accurate
    accuracy = accuracy_score(y_test, predicted_sentiments)
    print(f"Accuracy: {accuracy}")
    print(classification_report(y_test, predicted_sentiments))
    return X_train, X_test, y_train, y_test, train_ids, test_ids, model_ncf, indices


def get_recommendations(test_ids, X_train, y_train, model_ncf):
    recommendations = []
    for i, reviewer_id in enumerate(test_ids['reviewer_id'].unique()):
        idx = test_ids[test_ids['reviewer_id'] == reviewer_id].index
        if not idx.empty:
            similar_hikes = X_train.iloc[idx]
            predicted_sentiments = model_ncf.predict([similar_hikes['reviewer_id'], similar_hikes['hike_id']])
            highest_sentiment_hike_idx = np.argmax(predicted_sentiments)
            highest_sentiment_hike = X_train.iloc[highest_sentiment_hike_idx]
            recommendations.append((reviewer_id, highest_sentiment_hike))

    # Check recommendations
    print('Recommendations for each unique reviewer ID in the test set:')
    for reviewer_id, hike in recommendations:
        print(f'Reviewer ID: {reviewer_id}')
        print(f'Recommended hike: {hike}')









if __name__ == '__main__':
    print('Start Filtering')

    df_model = pd.read_csv(data_out + 'model_data1_no_pca.csv')
    #df_model['hik_id'] = df_model.reset_index(drop=False, inplace=False)# both should just be
    # columsn
    X_train, X_test, y_train, y_test, train_ids, test_ids, model_ncf = split_data_and_train(
        df_model)
    get_recommendations(test_ids, X_train, y_train, model_ncf)

    # PCA
    print('NOW WITH PCA')
    df_model_pca = pd.read_csv(data_out + 'model_data1_pca.csv')
    X_train, X_test, y_train, y_test, train_ids, test_ids, model_ncf = split_data_and_train(
        df_model_pca)
    get_recommendations(test_ids, X_train, y_train, model_ncf)



#POTETNIAL FUTURE IMPROVEMENTS: Make a time factor,where it only observes hike recommendations from
# same time of year?
#TODO: Finish this so it has accuracy, and complete a write up, discussing interpretation,
# implementation and visualization