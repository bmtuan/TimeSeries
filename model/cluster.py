from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import os

dir = 'data/multitask_train'
df_result = pd.DataFrame()
for path in os.listdir(dir):
    df = pd.read_csv(os.path.join(dir, path))
    df_result[path[:-4]] = df['PM2_5'].values[-40000:]

X = df_result.values.T
print(X.shape)
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
print(kmeans.labels_)
# for i in kmeans.labels_: 
#     print(i)
