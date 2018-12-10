from sklearn.metrics import make_scorer, accuracy_score 
from sklearn.model_selection import StratifiedShuffleSplit, GridSearchCV

from sklearn.linear_model import LogisticRegression

def classify(method, X_train, y_train, X_test, y_test):
    cv_sets = StratifiedShuffleSplit(n_splits = 2, test_size = 0.20, random_state = 5)
    cv_sets.get_n_splits(X_train, y_train)

    #Logistic regression
    if(method == 'lr'):
        print('Linear regression')
        logreg_clf = LogisticRegression()
        parameters_logreg = {"penalty": ["l2"], "fit_intercept": [True, False], "solver": ["newton-cg",
                             "lbfgs", "liblinear", "sag", "saga"], "max_iter": [50, 100, 200]}

        grid_logreg = GridSearchCV(logreg_clf, parameters_logreg, scoring=make_scorer(accuracy_score), 
                                   cv = cv_sets)
        grid_logreg.fit(X_train, y_train)

        logreg_clf = grid_logreg.best_estimator_

        logreg_clf.fit(X_train, y_train)
        y_pred = logreg_clf.predict(X_test)
        
        return y_pred, accuracy_score(y_test, y_pred)