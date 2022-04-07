from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

vectorizer = CountVectorizer(ngram_range=(1, 2))

documents = [
    'Ba ba ba ba',
    'A bloo bloo bloo',
    'waaaaa galawan',
]

X = vectorizer.fit_transform(documents)

print(vectorizer.get_feature_names_out())

print(X.toarray())

print(vectorizer.vocabulary_.get('waaaaa galawan'))

print(vectorizer.transform(['dewingo camma gando bim']).toarray())

feature_index = vectorizer.vocabulary_.get('bloo bloo')
print(X[:, feature_index].toarray())

transformer = TfidfTransformer(smooth_idf=False)

tfidf = transformer.fit_transform(X)

print("After fit_transform")

print(tfidf.toarray())

print(X.toarray())