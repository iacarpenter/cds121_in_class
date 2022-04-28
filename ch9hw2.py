import random
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

x1 = [random.uniform(-1,1) for _ in range(500)]
y1 = [random.uniform(-1,1) for _ in range(500)]
lab1 = [1 for _ in range(500)]

x2 = [random.uniform(3,6) for _ in range(500)]
y2 = [random.uniform(2,4) for _ in range(500)]
lab2 = [2 for _ in range(500)]

x3 = [random.uniform(2,4) for _ in range(500)]
y3 = [random.uniform(-1,-2) for _ in range(500)]
lab3 = [3 for _ in range(500)]

x = x1 + x2 + x3
y = y1 + y2 + y3
lab = lab1 + lab2 + lab3

data = pd.DataFrame(data={'x': x, 'y': y, 'label': lab})

# mix up the rows
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

labels_true = data.iloc[:, 2]
data.drop(data.columns[2], axis=1, inplace=True)

# Use the silhouette score to see whether the "best" number of clusters
# is as you would expect it to be

k_range = list(range(2,11))

for k in k_range:
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(data)
    score = silhouette_score(data, kmeans.labels_)
    print(f"{k} clusters silhouette score: {score:.5f}")

# the maximum silhouette score was achieved with 3 clusters, with a score of
# 0.72255, which as was expected in this case. However, these clusters are very
# cleanly seperated.