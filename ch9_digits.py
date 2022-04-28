from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

X_digits, y_digits = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits)

print(X_train.shape)
# Dimensions of original data: (1347, 64)
# 1347 images, 8 * 8 pixels

kmeans = KMeans(n_clusters=50)

X_transformed = kmeans.fit_transform(X_train)

print(X_transformed.shape)
# Dimensions of processed data: (1347, 50)
# 1347 instances, as many features as there are clusters

print(X_transformed)
# The data is transformed into an array where each instance has as many features
# as there are clusters (n_clusters) and the value of each feature is the distance
# of the instance from each cluster's centroid.
# When using KMeans for preprocessing, this transformed dataset is what we are
# training the model on.

pipeline = Pipeline([
    ("kmeans", KMeans()),
    ("log_reg", LogisticRegression(
        multi_class="ovr", 
        solver="lbfgs", 
        max_iter=5000, 
        random_state=42)),
])

param_grid = dict(kmeans__n_clusters=range(99, 151))
grid_clf = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy')
grid_clf.fit(X_train, y_train)

print("Best params:\n", grid_clf.best_params_)
# {'kmeans__n_clusters': 147}

print("Accuracy score:\n", grid_clf.score(X_test, y_test))
# 0.9866666666666667
# by increasing the number of clusters from 99 to 147 we increased the accuracy
# of our classifier from about 0.98222 to 0.98667