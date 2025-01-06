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
from tensorflow.keras.layers import Dropout
from sklearn.preprocessing import LabelBinarizer
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from keras.regularizers import l2, l1

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

    # OHE the target for loss function:
    lb = LabelBinarizer()
    y_train = lb.fit_transform(y_train)
    y_test = lb.transform(y_test)

    # model to get a baseline, and then make recommendations once accurate
    # This is a new model that I haven't used before, but I read through some tutorials.

    #---NCF Architecture: NCF uses embedding layers to represent users and items as dense vectors.
    # Input Layers: User and item IDs.
    # Embedding Layers: Learn user and item representations.
    # Concatenation Layer: Combine user and item embeddings.
    # Dense Layers: Learn complex user-item interactions.
    # Output Layer: Provide predictions

    #start by getting that dense vector
    user_input = Input(shape=(1,), name='user_input')
    item_input = Input(shape=(1,), name='item_input')
    user_embedding = Embedding(input_dim=df_model['reviewer_id'].nunique(), output_dim=16,
                               name='user_embedding')(user_input)
    item_embedding = Embedding(input_dim=df_model['hike_id'].nunique(), output_dim=16,
                               name='item_embedding')(item_input)

    user_vector = Flatten(name='user_vector')(user_embedding)
    item_vector = Flatten(name='item_vector')(item_embedding)
    merged = Concatenate()([user_vector, item_vector])

    from keras.layers import LeakyReLU
    #Then get the actual layers of the model
    # Use drop out layers to prevent overfitting, as initial model isn't imrpoving over epochss.
    dense_1 = Dense(32, activation='relu', kernel_regularizer=l2(0.07))(merged)
    dropout_1 = Dropout(0.6)(dense_1)


    # dense_2 = Dense(32, activation='tanh', kernel_regularizer=l2(0.002))(dropout_1) #
    # # kernel_regularizer=l2(0.01)
    # dropout_2 = Dropout(0.6)(dense_2)


    # dense_3 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(dropout_2)
    # dropout_3 = Dropout(0.6)(dense_3)

    #combine to get final fusion dense layer
    output = Dense(y_train.shape[1], activation='softmax')(dropout_1)

    model_ncf = Model(inputs=[user_input, item_input], outputs=output)

    #compile and fit, iterate, but early stop for overfitt. I can adjust learning rate here:
    model_ncf.compile(optimizer=Adam(learning_rate=0.0005), loss='categorical_crossentropy',
                      metrics=['accuracy'])


    early_stopping = EarlyStopping(monitor='val_loss', patience=5)
    # also lower learning rate to help with performance getting stuck:
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, min_lr=0.00001)
    model_ncf.fit([X_train['reviewer_id'], X_train['hike_id']], y_train, epochs=30, batch_size=32,
                  validation_split=0.2, callbacks=[early_stopping, reduce_lr], verbose=1)


    # get acc
    scores = model_ncf.evaluate([X_test['reviewer_id'], X_test['hike_id']], y_test, verbose=0)
    print(f"Accuracy: {scores[1]}")
    print(classification_report(np.argmax(y_test, axis=1),
                                np.argmax(model_ncf.predict([X_test['reviewer_id'],
                                                             X_test['hike_id']]), axis=1)))

    return X_train, X_test, y_train, y_test, train_ids, test_ids, model_ncf


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