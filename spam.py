import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score

DATASETS_PATH = "./datasets"

def load_spam_data(datasets_path=DATASETS_PATH):
    spam_csv_path = os.path.join(datasets_path, "spam_text.csv")
    return pd.read_csv(spam_csv_path, sep=",")

data = load_spam_data()

data["Category"]=data["Category"].replace({'spam':[0], 'ham':[1]})

def split_dataset(data, test_size=0.2, random_state=42):

    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in split.split(data, data["Category"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

    train_labels = strat_train_set["Category"].copy()
    strat_train_set = strat_train_set.drop(["Category"], axis=1)
    test_labels = strat_test_set["Category"].copy()
    strat_test_set = strat_test_set.drop(["Category"], axis=1)

    return strat_train_set, train_labels, strat_test_set, test_labels

train_set, train_labels, test_set, test_labels = split_dataset(data)

vectorizer = CountVectorizer()

train_set_prepared = vectorizer.fit_transform(train_set["Message"])

print("CountVectorizer:")
print(train_set_prepared[:10].toarray())
# Each unique word is 'tokenized' (given a unique integer as its identifyer). 
# Each individual document (each text) recieves a row, and each tokenized word
# recieves a column, so that there are as many columns as there are words with 
# 2 or more letters in all of the documents. The number in each row-column 
# intersection will be an integer representing how many times the word appeared 
# in each document. Most of these integeres will be 0, because each text only uses
# a small amount of the total words found in all of the documents.

transformer = TfidfTransformer()

tfidf = transformer.fit_transform(train_set_prepared)

print("TfidfTransformer:")
print(tfidf[0:10].toarray())
# This transformer replaces the integers from the CountVectorizer with 
# floating point numbers that 'weigh' each term to reflect the amount 
# of times the term appeared in the given document, the number of documents
# that contain the term, and the total number of documents. This is used to 
# shift emphasis away from very frequent terms and towards rarer terms in the 
# document set.

text_pipeline = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('transformer', TfidfTransformer()),
])

text_pipeline.fit(train_set)

lin_svc = LinearSVC()

lin_svc_scores = cross_val_score(
    estimator=lin_svc,
    X=train_set_prepared,
    y=train_labels,
    cv=3,
    scoring='accuracy')
print("LinearSVC scores:\n", lin_svc_scores)

sgd_clf = SGDClassifier()

sgd_clf_scores = cross_val_score(
    estimator=sgd_clf,
    X=train_set_prepared,
    y=train_labels,
    cv=3,
    scoring='accuracy')
print("SGDClassifier scores:\n", sgd_clf_scores)

tree_clf = DecisionTreeClassifier()

tree_clf_scores = cross_val_score(
    estimator=tree_clf,
    X=train_set_prepared,
    y=train_labels,
    cv=3,
    scoring='accuracy')
print("DecisionTreeClassifier scores:\n", sgd_clf_scores)

# I chose to go forward with LinearSVC because its accuracy score
# appears to be consistently higher than for the other two, though
# by a very small margin. The better accuracy score means that the 
# classifier's predictions matched the real labels more often.

lin_svc_param_grid = [
    {'loss': ['squared_hinge', 'hinge'], 'C': [1, 10, 100],
    'max_iter': [10_000]}
]
# added max_iter of 10_1000 because it failed to converge at the default of 1000

grid_search = GridSearchCV(lin_svc, lin_svc_param_grid, cv=3,
                            scoring='accuracy', return_train_score=True)

grid_search.fit(train_set_prepared, train_labels)
print(grid_search.best_params_)
# {'C': 1, 'loss': 'squared_hinge', 'max_iter': 10000}

final_lin_svc = LinearSVC(C=1, loss='squared_hinge', max_iter=10000)

final_lin_svc.fit(train_set_prepared, train_labels)

test_set_prepared = text_pipeline.transform(test_set["Message"])

final_predictions = final_lin_svc.predict(test_set_prepared)

final_accuracy = accuracy_score(test_labels, final_predictions)
print("Final accuracy score:", final_accuracy)


# ValueError: X has 1 features, but LinearSVC is expecting 7736 features as input.