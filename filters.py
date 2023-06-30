import math
from classification import train_decisionstump_auc
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import StratifiedKFold #maintain class proportions when creating folds
import classification as cls
import auxiliary_functions as aux
from numpy import median
from numpy import var


# returns a sorted dict object with all features as keys and their scores as values for the decision stump filter
def getscores_decisionstumpfilter(df):
    feature_score_dict = {}
    for feature in df.columns[:-1]:  # all features but the Class (last one)
        selected_features = [feature, df.columns[-1]]
        stumpdf = df.filter(selected_features)
        feature_score_dict[feature] = train_decisionstump_auc(stumpdf)
    return dict(sorted(feature_score_dict.items(), key=lambda item: item[1]))


# returns a sorted dict object with all features as keys and their scores as values for the Log Odds Ratio filter
def getscores_logoddsfilter(df):
    feature_score_dict = {}
    class_array = df.iloc[:, -1].values
    n_instances = df.shape[0]
    for fpos in range(0, df.shape[1]-1):
        feature_name = df.columns[fpos]
        feature_array = df.iloc[:, fpos].values
        count00 = 0
        count01 = 0
        count10 = 0
        count11 = 0
        for pos in range(0, n_instances):
            if feature_array[pos] == 0 and class_array[pos] == 0:
                count00 += 1
            elif feature_array[pos] == 0 and class_array[pos] == 1:
                count01 += 1
            elif feature_array[pos] == 1 and class_array[pos] == 0:
                count10 += 1
            elif feature_array[pos] == 1 and class_array[pos] == 1:
                count11 += 1
        if count00 < 5 or count10 < 5 or count01 < 5 or count11 < 5:  # correction to avoid values too low
            oddsratio = ((count00+0.5) * (count11+0.5)) / ((count10+0.5) * (count01+0.5))
        else:
            oddsratio = (count00*count11)/(count10*count01)
        logoddsratio = math.fabs(math.log10(oddsratio))
        feature_score_dict[feature_name] = logoddsratio
    return dict(sorted(feature_score_dict.items(), key=lambda item: item[1]))


# returns a sorted dict object with all features as keys and their scores as values for the AsymmetricOptimalPrediction
def getscores_asymmetricfilter(df):
    feature_score_dict = {}
    class_array = df.iloc[:, -1].values
    n_instances = df.shape[0]
    count_pos = 0
    for fpos in range(0, df.shape[1]-1):
        feature_name = df.columns[fpos]
        feature_array = df.iloc[:, fpos].values
        count00 = 0
        count01 = 0
        count10 = 0
        count11 = 0
        for pos in range(0, n_instances):
            if feature_array[pos] == 0 and class_array[pos] == 0:
                count00 += 1
            elif feature_array[pos] == 0 and class_array[pos] == 1:
                count01 += 1
            elif feature_array[pos] == 1 and class_array[pos] == 0:
                count10 += 1
            elif feature_array[pos] == 1 and class_array[pos] == 1:
                count11 += 1
        perc00 = count00 / df.shape[0]
        perc01 = count01 / df.shape[0]
        perc10 = count10 / df.shape[0]
        perc11 = count11 / df.shape[0]
        if (count00 + count10) > (count01 + count11):
            error_case1 = 1 - (perc00 + perc10)  # predict B1 regardless of A value
        else:
            error_case1 = 1 - (perc01 + perc11)  # predict B2 regardless of A value
        error_case2 = 1 - (max(perc00, perc01) + max(perc10, perc11))
        if error_case1 != error_case2:
            count_pos += 1
        feature_score_dict[feature_name] = (error_case1 - error_case2) / error_case1
    return dict(sorted(feature_score_dict.items(), key=lambda item: item[1]))


# Combines the AOP and Log Odds filter score calculations in a single function, to optimise the runtime.
# returns returns two sorted dict objects (AOP, LogOdds) with all features as keys and their scores as values
def getscores_asymmetricfilter_logodds(df):
    aop_score_dict = {}
    logoddsratio_score_dict = {}
    class_array = df.iloc[:, -1].values
    n_instances = df.shape[0]
    count_pos = 0
    for fpos in range(0, df.shape[1]-1):
        feature_name = df.columns[fpos]
        feature_array = df.iloc[:, fpos].values
        count00 = 0
        count01 = 0
        count10 = 0
        count11 = 0
        for pos in range(0, n_instances):
            if feature_array[pos] == 0 and class_array[pos] == 0:
                count00 += 1
            elif feature_array[pos] == 0 and class_array[pos] == 1:
                count01 += 1
            elif feature_array[pos] == 1 and class_array[pos] == 0:
                count10 += 1
            elif feature_array[pos] == 1 and class_array[pos] == 1:
                count11 += 1
        perc00 = count00 / df.shape[0]
        perc01 = count01 / df.shape[0]
        perc10 = count10 / df.shape[0]
        perc11 = count11 / df.shape[0]
        if (count00 + count10) > (count01 + count11):
            error_case1 = 1 - (perc00 + perc10)  # predict B1 regardless of A value
        else:
            error_case1 = 1 - (perc01 + perc11)  # predict B2 regardless of A value
        error_case2 = 1 - (max(perc00, perc01) + max(perc10, perc11))
        if error_case1 != error_case2:
            count_pos += 1
        aop_score_dict[feature_name] = (error_case1 - error_case2) / error_case1

        if count00 < 5 or count10 < 5 or count01 < 5 or count11 < 5:
            oddsratio = ((count00+0.5) * (count11+0.5)) / ((count10+0.5) * (count01+0.5))
        else:
            oddsratio = (count00*count11)/(count10*count01)
        logoddsratio = math.fabs(math.log10(oddsratio))
        logoddsratio_score_dict[feature_name] = logoddsratio
    return dict(sorted(aop_score_dict.items(), key=lambda item: item[1])), dict(sorted(logoddsratio_score_dict.items(), key=lambda item: item[1]))


# returns a SORTED dict object with all features as keys and their scores as values
# to apply the filter, we selected the 'k' last elements (the highest scores) of that list of features in the filter
def calculate_scores_nonnativefilter(df, filter_method):
    if filter_method == "DecisionStump":
        return getscores_decisionstumpfilter(df)
    elif filter_method == "LogOddsRatio":
        return getscores_logoddsfilter(df)
    elif filter_method == "AsymmetricOptimalPrediction":
        return getscores_asymmetricfilter(df)
    elif filter_method == "AsymmetricOptimalPrediction_LogOddsRatio":
        return getscores_asymmetricfilter_logodds(df)
    else:
        print(f'Error: unexpected filter method: {filter_method}.')
        print("Exiting the program...")
        exit(0)


# selects the last 'nfeatures' columns in the dataframe, based on a dict object assigning a score to each feature
# note that the Class feature is included in the filtered dataset by manually adding the last feature to the list
def apply_filter(df, feature_score_dict, nfeatures):
    selected_features = list(feature_score_dict)[-(nfeatures):]
    selected_features.append(df.columns[-1]) #add Class label as the last feature (assumes class in last position!)
    return df.filter(selected_features)


# Creates a filtered version of X_train and X_test (feature values of a dataframe, used to train the classifier)
# By applying the selected filter method and nfeatures (k value)
def apply_chosenfilterstrategy(df, train_index, test_index, filter_method, nfeatures):
    if nfeatures > df.shape[1] or nfeatures <=0:
        print(f'Warning, invalid k value. k: {nfeatures} features: {df.shape[1]}. Setting k as all.')
        nfeatures = 'all'
    else:
        print(f'Applying filter {filter_method} with k: {nfeatures}')
    X_train, X_test = df.iloc[train_index, :-1], df.iloc[test_index, :-1]
    y_train = df.iloc[train_index, -1]
    if filter_method in ['DecisionStump', 'LogOddsRatio', 'AsymmetricOptimalPrediction']:  # Non-native methods
        feature_score_dict = calculate_scores_nonnativefilter(df.iloc[train_index, :], filter_method)
        filtereddf = apply_filter(df, feature_score_dict, nfeatures)
        return filtereddf.iloc[train_index, :-1], filtereddf.iloc[test_index, :-1]  # filtered X_train and X_test
    else:  # Native filter method, InfoGain or Chi2
        if filter_method == "InfoGain":
            filter_method = mutual_info_classif  # Information Gain filter
        elif filter_method == "Chi2":
            filter_method = chi2  # Information Gain filter
        else:
            print(f'Error. Unexpected feature method name: {filter_method}.')
            print("Exiting the program...")
            exit(0)
        filter = SelectKBest(filter_method, k=nfeatures)
        filter.fit(X_train, y_train)
        return filter.transform(X_train), filter.transform(X_test)  # filtered X_train and X_test


# native filter, need to be run once for every candidate nfeatures
def runNativeFilter(fmethod, nfeatures, X, y, apply_undersampling):
    score_array = []  # creates a new array before each internalCV loop of training classifiers, so it's empty
    kf = StratifiedKFold(n_splits=5, shuffle=False)
    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        if fmethod == "InfoGain":
            filter_method = mutual_info_classif
        elif fmethod == "Chi2":
            filter_method = chi2
        filter_alg = SelectKBest(filter_method, k=nfeatures)
        filter_alg.fit(X_train, y_train)
        X_train = filter_alg.transform(X_train)
        X_test = filter_alg.transform(X_test)
        score_array.append(cls.train_randomforest_AUC(X_train, X_test, y_train, y_test, apply_undersampling))
        #score_array.append(cls.train_randomforest_gmean(X_train, X_test, y_train, y_test, apply_undersampling)) # for GMean scoring
    return {(fmethod, nfeatures): {"score": median(score_array), "variance": var(score_array)}}


 # non-native filters, can get score only once and apply to each candidate nfeatures
def runNonNativeFilter(fmethod, nfeatures_array, df, apply_undersampling):
    X = df.iloc[:, :-1]  # table excluding the class column
    y = df.iloc[:, -1]  # only the class column
    kf = StratifiedKFold(n_splits=5, shuffle=False)
    score_arrays = [[] for _ in range(len(nfeatures_array))]

    for train_index, test_index in kf.split(X, y):
        feature_score_dict = calculate_scores_nonnativefilter(df.iloc[train_index, :], fmethod)
        for i in range(len(nfeatures_array)):
            filtereddf = apply_filter(df, feature_score_dict, nfeatures_array[i])
            X_train, X_test = filtereddf.iloc[train_index, :-1], filtereddf.iloc[test_index, :-1]
            y_train, y_test = filtereddf.iloc[train_index, -1], filtereddf.iloc[test_index, -1]
            score_arrays[i].append(cls.train_randomforest_AUC(X_train, X_test, y_train, y_test, apply_undersampling))
            #score_arrays[i].append(cls.train_randomforest_gmean(X_train, X_test, y_train, y_test, apply_undersampling)) #for GMean scoring

    local_candidate_filters = {}
    for i in range(len(score_arrays)):
        local_candidate_filters.update({(fmethod, nfeatures_array[i]): {"score": median(score_arrays[i]), "variance": var(score_arrays[i])}})
    return local_candidate_filters


def runAutoFilter(df, apply_undersampling, nfeatures_array):
    X = df.iloc[:, :-1]  # set outside the for loop to save some time in runNativeFilter()
    y = df.iloc[:, -1]
    candidate_filters = aux.create_candidate_filters_dict(nfeatures_array)
    for fmethod, nfeatures in candidate_filters:
        print(f'Running InternalCV for AutoFilter. Method: {fmethod} k: {nfeatures}')
        if fmethod in ['InfoGain', 'Chi2']:
            candidate_filters.update(runNativeFilter(fmethod, nfeatures, X, y, apply_undersampling))
        elif fmethod in ['DecisionStump', 'LogOddsRatio', 'AsymmetricOptimalPrediction']:
            candidate_filters.update(runNonNativeFilter(fmethod, nfeatures_array, df, apply_undersampling))
        else:
            print(f'Error, unexpected candidate filter method: {fmethod}')
            print("Exiting the program...")
            exit(0)
    aux.printdict(candidate_filters)
    return aux.select_autofilter_strategy(candidate_filters)

