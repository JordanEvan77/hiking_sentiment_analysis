import sklearn
import pandas as pd
import numpy as np
from sklearn.neighbors import NearestNeighbors

df_model = pd.read_csv('data\model_ready\model_data1.csv')

# reshape data into numyp array:
piv = df_model.pivot_table(index='Trail Report By', columns='Hike Name', values=).fillna(0)

x_cols = []
X = piv[x_cols].to_numpy()
Y = piv[['Sentiment']]

#rough outline:
knn = NearestNeighbors(metric='cosine', algorithm='brute')
knn.fit(X)


# TODO: DO A MORE SPECIFIC FILTERING ALGO TO COMPARE


if __name__ == '__main__':
    print('Start Filtering')
