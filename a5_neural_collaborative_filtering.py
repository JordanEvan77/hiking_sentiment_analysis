# THIS IS JUST A SLIGHT ADJUSMENT FROM THE A4 script so that I can test out the NCF algorithm and
# its accuracy and recommendations

import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate

df_model = pd.read_csv('data\model_ready\model_data1.csv')


hike_attributes = [i for i in df_model.columns if i not in ['sentiment']]
X = df_model[hike_attributes]
y = df_model[['sentiment']]
reviewer_ids = df_model[['reviewer_id']]

X_train, X_test, y_train, y_test, train_ids, test_ids = train_test_split(X, y, reviewer_ids,
                           test_size=0.2, random_state=42)

num_reviewers = df_model['reviewer_id'].nunique()
num_hike_features = X_train.shape[1]

#create general input for the model
input_reviewer = Input(shape=(1,), name='reviewer')
input_hike = Input(shape=(num_hike_features,), name='hike_features')

#create layers and embed
embedding_reviewer = Embedding(num_reviewers, 8, input_length=1)(input_reviewer)
flatten_reviewer = Flatten()(embedding_reviewer)

concatenated = Concatenate()([flatten_reviewer, input_hike])
dense1 = Dense(128, activation='relu')(concatenated)
dense2 = Dense(64, activation='relu')(dense1)
output = Dense(1, activation='sigmoid')(dense2)

model = Model([input_reviewer, input_hike], output)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the NCF model
model.fit([train_ids, X_train], y_train, epochs=10, batch_size=32, validation_split=0.2)

# check acc
y_pred = model.predict([test_ids, X_test])
y_pred_binary = (y_pred > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_binary)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred_binary))

#recommendations generate
recommendations = []
for i, reviewer_id in enumerate(test_ids['reviewer_id_encoded'].unique()):
    idx = test_ids[test_ids['reviewer_id_encoded'] == reviewer_id].index
    if not idx.empty:
        reviewer_input = np.array([reviewer_id])
        hike_predictions = model.predict([reviewer_input, X_test])
        highest_sentiment_hike_idx = np.argmax(hike_predictions)
        highest_sentiment_hike = X_test.iloc[highest_sentiment_hike_idx]
        recommendations.append((reviewer_id, highest_sentiment_hike))


print('Recommendations for each unique reviewer ID in the test set:')
for reviewer_id, hike in recommendations:
    print(f'Reviewer ID: {reviewer_id}')
    print(f'Was reccomended hike: {hike}')




if __name__ == '__main__':
    print('Start Neural Filtering')