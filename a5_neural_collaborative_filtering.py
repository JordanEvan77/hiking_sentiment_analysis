import sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import accuracy_score, classification_report
from Scratch.loggers import data_dir, data_out
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Flatten, Dense, Concatenate, Add
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2, l1
from keras.layers import BatchNormalization

def split_data_and_format(df_model):
    hike_attributes = [i for i in df_model.columns if i not in ['sentiment', 'hike_id',
                                                                'reviewer_id']]
    id_attributes = ['hike_id', 'reviewer_id']
    X = df_model[hike_attributes+id_attributes]
    df_model['sentiment'] = df_model['sentiment'] + 1 #adjust to make it in range for model
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

    # OHE the target for loss function:
    # lb = LabelBinarizer()
    # y_train = lb.fit_transform(y_train)
    # y_test = lb.transform(y_test)

    # model to get a baseline, and then make recommendations once accurate
    # This is a new model that I haven't used before, but I read through some tutorials.

    #---NCF Architecture: NCF uses embedding layers to represent users and items as dense vectors.
    # Input Layers: user and item IDs.
    # Embedding Layers: learn user and item representations.
    # Concatenation layer: Combine user and item embeddings.
    # Dense Layers: elarn complex user-item interactions.
    # Output Layer: provide predictions

    #start by getting that dense vector
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')
    hike_info_input = Input(shape=(len(hike_attributes,)))

    user_embedding = Embedding(input_dim=df_model['reviewer_id'].nunique(), output_dim=16,
                               name='user_embedding')(user_input)
    item_embedding = Embedding(input_dim=df_model['hike_id'].nunique(), output_dim=16,
                               name='item_embedding')(item_input)

    user_vector = Flatten(name='user_vector')(user_embedding)
    item_vector = Flatten(name='item_vector')(item_embedding)

    merged = Concatenate()([user_vector, item_vector])
    merged_with_info = Concatenate()([merged, hike_info_input])

    return X_train, X_test, y_train, y_test, train_ids, test_ids, merged_with_info, user_input,\
           item_input, hike_info_input, hike_attributes



def model_build_and_test(X_train, X_test, y_train, y_test, train_ids, test_ids, merged_with_info,
                         user_input, item_input, hike_info_input, hike_attributes):
    #Then get the actual layers of the model
    # Use drop out layers to prevent overfitting, as initial model isn't imrpoving over epochss.
       # Dense layers
    dense1 = Dense(128, activation='relu')(merged_with_info)
    batch_norm2 = BatchNormalization()(dense1)
    dropout1 = Dropout(0.5)(batch_norm2)

    dense2 = Dense(64, activation='relu')(dropout1)
    batch_norm2 = BatchNormalization()(dense2)
    dropout2 = Dropout(0.5)(batch_norm2)

    dense3 = Dense(64, activation='relu')(dropout2)
    batch_norm3 = BatchNormalization()(dense3)
    dropout3 = Dropout(0.5)(batch_norm3)

    residual = Add()([dropout3, dropout2])

    output = Dense(3, activation='softmax')(residual)  # corrected to3 classes:0,1,2

    model_ncf = Model(inputs=[user_input, item_input, hike_info_input], outputs=output)
    model_ncf.compile(optimizer=Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])
    #
    early_stopping = EarlyStopping(monitor='val_loss', patience=6)
    # # also lower learning rate to help with performance getting stuck:
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)
    model_ncf.fit([X_train['reviewer_id'], X_train['hike_id'], X_train[hike_attributes]], y_train, epochs=30,
                  batch_size=32,
                  validation_split=0.2, verbose=1, callbacks=[early_stopping, reduce_lr])

    # get acc
    scores = model_ncf.evaluate([X_test['reviewer_id'], X_test['hike_id'],
                                 X_test[hike_attributes]], y_test, verbose=0)
    print(f"Test Accuracy: {scores[1]}")
    print(classification_report(np.argmax(y_test, axis=1),
                                np.argmax(model_ncf.predict([X_test['reviewer_id'],
                                                             X_test['hike_id'],
                                                             X_test[hike_attributes]]), axis=1)))
    return model_ncf, hike_attributes



def get_recommendations(test_ids, X_train, y_train, model_ncf, hike_attributes):
    recommendations = []
    for i, reviewer_id in enumerate(test_ids['reviewer_id'].unique()):
        idx = test_ids[test_ids['reviewer_id'] == reviewer_id].index
        if not idx.empty:
            similar_hikes = X_train.iloc[idx]
            predicted_sentiments = model_ncf.predict([similar_hikes['reviewer_id'],
                                                      similar_hikes['hike_id'],
                                                      similar_hikes[hike_attributes]])

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
    X_train, X_test, y_train, y_test, train_ids, test_ids, merged_with_info, user_input, \
    item_input, hike_info_input, hike_attributes = split_data_and_format(df_model)

    model_ncf, hike_attributes = model_build_and_test(X_train, X_test, y_train, y_test, train_ids,
                    test_ids, merged_with_info, user_input, item_input, hike_info_input, hike_attributes)

    get_recommendations(test_ids, X_train, y_train, model_ncf, hike_attributes)


    # PCA
    print('NOW WITH PCA')
    df_model_pca = pd.read_csv(data_out + 'model_data1_pca.csv')

    X_train, X_test, y_train, y_test, train_ids, test_ids, merged_with_info, user_input, \
    item_input, hike_info_input = split_data_and_format(df_model_pca)

    model_ncf, hike_attributes = model_build_and_test(X_train, X_test, y_train, y_test, train_ids,
                        test_ids, merged_with_info, user_input, item_input, hike_info_input)

    get_recommendations(test_ids, X_train, y_train, model_ncf, hike_attributes)


#POTETNIAL FUTURE IMPROVEMENTS: Make a time factor,where it only observes hike recommendations from
# same time of year?