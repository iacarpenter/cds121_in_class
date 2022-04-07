from sklearn.datasets import make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

X, y = make_circles(n_samples=1000, factor=0.4, random_state=0, noise=0.05)
X_train, X_test, y_train, y_test  = train_test_split(X, y, stratify=y, random_state=0)

polynomial_svc = Pipeline([
    ("svc", SVC(kernel='poly', degree=2, coef0=1))
])

polynomial_svc.fit(X_train, y_train)

svc_scores = cross_val_score(
    estimator=polynomial_svc,
    X=X_train,
    y=y_train,
    cv=3,
    scoring='accuracy')
print("Cross-val scores:\n", svc_scores)

y_train_pred = cross_val_predict(polynomial_svc, X_train, y_train, cv=3)
cm = confusion_matrix(y_train, y_train_pred)
print("Confusion matrix:")
print(cm)

