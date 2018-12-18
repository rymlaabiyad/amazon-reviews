from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier
from treeinterpreter import treeinterpreter as ti
from sklearn.preprocessing import scale
import numpy as np


def classify(method, X_train, y_train, X_test, y_test):
    cv_sets = StratifiedShuffleSplit(
        n_splits=2, test_size=0.20, random_state=5)
    cv_sets.get_n_splits(X_train, y_train)

    # Logistic regression
    if(method == 'lr'):
        print('Linear regression')
        logreg_clf = LogisticRegression()
        parameters_logreg = {"fit_intercept": [True, False], "solver": ["newton-cg",
                                                                        "lbfgs",
                                                                        "liblinear", "sag", "saga"], "max_iter": [50, 100, 200]}

        grid_logreg = GridSearchCV(logreg_clf, parameters_logreg, scoring=make_scorer(accuracy_score),
                                   cv=cv_sets)
        grid_logreg.fit(X_train, y_train)
        best_clf = grid_logreg.best_estimator_
        best_clf.fit(X_train, y_train)
        y_pred = best_clf.predict(X_test)

    elif(method == 'svc'):
        print('Support Vectors Classifier')
        svc_clf = SVC()
        parameters_svc = {"kernel": ["linear"], "probability": [True, False]}
        grid_svc = GridSearchCV(svc_clf, parameters_svc,
                                scoring=make_scorer(accuracy_score), cv=cv_sets)
        grid_svc.fit(X_train, y_train)
        best_clf = grid_svc.best_estimator_
        best_clf.fit(X_train, y_train)
        y_pred = best_clf.predict(X_test)

    elif(method == 'dt'):
        print('Decision Tree')
        DT_clf = DecisionTreeClassifier()
        parameters_dt = {"max_depth": [5, 6, 7, 10], "min_samples_split": [2, 3, 5, 10],
                         "max_features": ["auto", "sqrt", "log2"],
                         "criterion": ["gini"]}
        grid_dt = GridSearchCV(DT_clf, parameters_dt,
                               scoring=make_scorer(accuracy_score), cv=cv_sets)
        grid_dt.fit(X_train, y_train)
        best_clf = grid_dt.best_estimator_
        best_clf.fit(X_train, y_train)
        y_pred = best_clf.predict(X_test)

    elif(method == 'nb'):
        print('Naive Bayes Classifier')
        best_clf = GaussianNB()
        best_clf.fit(X_train, y_train)
        y_pred = best_clf.predict(X_test)
    return best_clf, accuracy_score(y_test, y_pred)


def featureanalysis(features, clf):

    if(clf.__class__.__name__ == 'DecisionTreeClassifier'):
        prediction, bias, contributions = ti.predict(clf, features)
        totalc = np.mean(contributions, axis=0)
        totalc0 = totalc[:, 0]
        totalc1 = totalc[:, 1]
        feature_to_coef3 = {
            word: coef for word, coef in zip(
                features.columns, totalc0
            )
        }
        for best_positive in sorted(
                feature_to_coef3.items(),
                key=lambda x: x[1],
                reverse=True)[:5]:
            print (best_positive)
        feature_to_coef4 = {
            word: coef for word, coef in zip(
                features.columns, totalc1
            )
        }
        for best_negative in sorted(
                feature_to_coef4.items(),
                key=lambda x: x[1], reverse=True)[:5]:
            print (best_negative)
    else:
        feature_to_coef = {
            word: coef for word, coef in zip(
                features.columns, clf.coef_[0]
            )
        }
        print('The most liked features are :')
        for best_positive in sorted(
                feature_to_coef.items(),
                key=lambda x: x[1],
                reverse=True)[:5]:
            print (best_positive)

        print('\n' + 'The most disliked features are : ')
        for best_negative in sorted(
                feature_to_coef.items(),
                key=lambda x: x[1])[:5]:
            print (best_negative)
