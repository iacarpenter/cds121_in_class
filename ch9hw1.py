import random
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import homogeneity_score, completeness_score, v_measure_score

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

k = 3
kmeans = KMeans(n_clusters=k)
y_pred = kmeans.fit_predict(data)

print("Centroids:\n", kmeans.cluster_centers_)

print("Homogeneity:\n", homogeneity_score(labels_true, y_pred))
print("Completeness:\n", completeness_score(labels_true, y_pred))
print("V-measure:\n", v_measure_score(labels_true, y_pred))

# all perfect scores of 1.0