import os
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

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

X = vectorizer.fit_transform(train_set["Message"])

print("CountVectorizer:")
print(X[:10].toarray())
# Each unique word is 'tokenized' (given a unique integer as its identifyer). 
# Each individual document (each text) recieves a row, and each tokenized word
# recieves a column, so that there are as many columns as there are words with 
# 2 or more letters in all of the documents. The number in each row-column 
# intersection will be an integer representing how many times the word appeared 
# in each document. Most of these integeres will be 0, because each text only uses
# a small amount of the total words found in all of the documents.

transformer = TfidfTransformer()

tfidf = transformer.fit_transform(X)

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

test_set_prepared = text_pipeline.fit_transform(test_set["Message"])

print("Prepared test set:")
print(test_set_prepared[:10].toarray())