from wine_functions import fetch_wine_data, load_red_wine_data, load_white_wine_data, \
    add_color_feature, concat_dataframes, split_dataset
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score

# fetch_wine_data()

red_wine = load_red_wine_data()
white_wine = load_white_wine_data()
add_color_feature(red_wine, white_wine)
wine = concat_dataframes(red_wine, white_wine)

# Split off 'color' labels
train_set, train_labels, test_set, test_labels = split_dataset(wine)

# Linear support vector machine classifier:

'''
lin_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("linear_svc", LinearSVC()),
])

lin_svm_clf.fit(train_set, train_labels)

lin_svm_clf_scores = cross_val_score(
    estimator=lin_svm_clf,
    X=train_set,
    y=train_labels,
    cv=3,
    scoring='accuracy')

print("Scores:\n", lin_svm_clf_scores)
# ConvergenceWarning: Liblinear failed to converge,
# increase the number of iterations.
'''

# Nonlinear support vector machine classifier

svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svc", SVC()),
])

svm_clf.fit(train_set, train_labels)

svm_clf_scores = cross_val_score(
    estimator=svm_clf,
    X=train_set,
    y=train_labels,
    cv=3,
    scoring='accuracy')

print("Scores:\n", svm_clf_scores)
# [0.99480669 0.9965358  0.9948037 ]