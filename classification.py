from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.model_selection import StratifiedKFold #maintain class proportions when creating folds
from numpy import median
from math import sqrt


# Trains a RF classifier given a train and test set and returns its area under the roc curve result
def train_randomforest_AUC(X_train, X_test, y_train, y_test, apply_undersampling):
    if apply_undersampling:
        rf = BalancedRandomForestClassifier(n_estimators=500, random_state=0)
    else:
        rf = RandomForestClassifier(n_estimators=500, random_state=0, class_weight='balanced_subsample')
    rf = rf.fit(X_train, y_train)
    y_pred_prob = rf.predict_proba(X_test)
    AUC = roc_auc_score(y_test, y_pred_prob[:, 1])
    return AUC


# Trains a RF classifier given a train and test set and returns its geometric mean result
def train_randomforest_gmean(X_train, X_test, y_train, y_test, apply_undersampling):
    if apply_undersampling:
        rf = BalancedRandomForestClassifier(n_estimators=500, random_state=0)
    else:
        rf = RandomForestClassifier(n_estimators=500, random_state=0)
    rf = rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    if tp == 0 or tn == 0:
        return 0
    sensitivity = tp/(tp+fp)
    specificity = tn/(tn+fn)
    return sqrt(sensitivity*specificity)


# Performs a 5-fold cross-validation with a dataset, training a decision stump for each fold. Returns median AUC
def train_decisionstump_auc(df):
    X = df.iloc[:, :-1]  # table excluding the class column, used to get test_set (and training when not undersampling)
    y = df.iloc[:, -1]  # only the class column, used to get test_set (and training when not undersampling)
    kf = StratifiedKFold(n_splits=5, shuffle=False)
    score_array = []
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        ds = tree.DecisionTreeClassifier()
        ds = ds.fit(X_train, y_train)
        y_pred_prob = ds.predict_proba(X_test)
        score_array.append(roc_auc_score(y_test, y_pred_prob[:, 1]))
    return median(score_array)

