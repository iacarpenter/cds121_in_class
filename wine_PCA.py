from wine_functions import load_red_wine_data, load_white_wine_data, add_color_feature, \
    concat_dataframes, split_dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score

red_wine = load_red_wine_data()
white_wine = load_white_wine_data()
add_color_feature(red_wine, white_wine)
wine = concat_dataframes(red_wine, white_wine)

# splitting the color off as the labels
# using stratified sampling according to the color category
train, train_labels, test, test_lables = split_dataset(wine)

pipeline = Pipeline([
    ("std_scaler", StandardScaler()),
])

train_prepared = pipeline.fit_transform(train)

pca = PCA(n_components = 2)
train_2D = pca.fit_transform(train_prepared)
print(pca.explained_variance_ratio_)
# [0.25344983 0.22047089]
# we preserved only 47% of the variance of the data with 2 dimensions

lin_svc = LinearSVC(max_iter=10000)

lin_svc.fit(train_2D, train_labels)

lin_svc_scores = cross_val_score(
    estimator=lin_svc,
    X=train_2D,
    y=train_labels,
    cv=3,
    scoring='accuracy')

print("Scores:\n", lin_svc_scores)
# [0.97980381 0.98267898 0.98498845]
# It predicts the color with a high level of accuracy