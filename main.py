from numpy import median
from sklearn.model_selection import StratifiedKFold  # maintains class proportions when creating folds
import auxiliary_functions as aux
from classification import train_randomforest_AUC
from filters import runAutoFilter
from filters import apply_chosenfilterstrategy


# Creates a train and test set for each fold of a stratified kfold process,
# then calls the train_classifier function and prints the average results
def runClassificationExperiment(df, apply_undersampling, unusable_feature_threshold, filter_method, nfeatures):
    kf = StratifiedKFold(n_splits=10, shuffle=False)
    score_array = []
    fold = 0
    for train_index, test_index in kf.split(df.iloc[:, :-1], df.iloc[:, -1]):
        fold = fold + 1
        training_set = df.iloc[train_index, :]
        test_set = df.iloc[test_index, :]
        training_set, test_set = aux.removeLowFrequencyFeatures_TrainTest(training_set, test_set, unusable_feature_threshold)

        if filter_method == "None":
            X_train, X_test = training_set.iloc[:, :-1], test_set.iloc[:, :-1]
        elif filter_method == "AutoFilter":
            nfeatures_array = aux.set_nfeatures_array(df.shape[1])  # k values to be used (GenAge dataset)
            filter_method, nfeatures = runAutoFilter(training_set, apply_undersampling, nfeatures_array)
            print(f'AutoFilter InternalCV end. Selected filter strategy : {filter_method} with k = {nfeatures}')
            X_train, X_test = apply_chosenfilterstrategy(df, train_index, test_index, filter_method, nfeatures)
        else:  # other filter methods
            X_train, X_test = apply_chosenfilterstrategy(df, train_index, test_index, filter_method, nfeatures)

        y_train, y_test = training_set.iloc[:, -1], test_set.iloc[:, -1]

        #Training classifier - AUC Scoring
        score = train_randomforest_AUC(X_train, X_test, y_train, y_test, apply_undersampling)
        #score = train_randomforest_gmean(X_train, X_test, y_train, y_test, apply_undersampling) #for GMean scoring
        score_array.append(score)
        print(f"Fold {fold}. AUC score: {round(score, 3)}")

    print(f"Median AUC: {round(median(score_array),3)}")


"""
Main function: runs a 10-fold cross-validation and returns the AUC results of Random Forest classifiers trained at each fold, and the median AUC value.
The input datasets should be formatted as tab-separated spreadsheets with an index column as the first (leftmost) variable, and a binary (0,1) class value in the last (rightmost) position
For our test datasets, there are some additional identifier variables which are removed when loading the dataset: STITCH_Code, InteractorsList, InteractorsCount
The following experiment parameters that can be set by the user:

unusable_feature_threshold: 
    Features with too few different values are removed in a preprocessing step, removing spurious variables
    This works well for binary features, but is not recommended for features with numerical or categorical values
    int value (recommended for our test datasets: 10). If set to a value < 0, the threshold filter is not applying
    
BRF_undersampling:
    Whether the classifiers should be trained using balanced datasets to avoid a bias in favour of the majority class
    True: undersample training set to a 1:1 ratio (uses BalancedRandomForestClassifier from imblearn.ensemble library).
    False: trains classifier with unchanged class balance (Not recommended for imbalanced datasets)
    
filter_method: 
    Selected filter scoring strategy. Accepted values: 
    "None": Do not apply a filter to the dataset, and use all predictive features when training the classifier
    "AutoFilter": Select a candidate filter and k value by testing all combinations in an internal cross-validation
    "InfoGain": Apply Information Gain filter (uses native implementation from sklearn)
    "Chi2": Apply Chi2 filter (uses native implementation from sklearn)
    "DecisionStump": Apply Decision Stump filter (implemented in this project)
    "LogOddsRatio": Apply Log Odds Ratio filter (implemented in this project)
    "AsymmetricOptimalPrediction": Apply Asymmetric Optimal Prediction filter (implemented in this project)
    
nfeatures: number of predictive features to be selected when applying the filter method
    int value, irrelevant for filter_method = None (no filter) and for filter_method = AutoFilter (chosen automatically)    
"""
def main(filepath, BRF_undersampling, unusable_feature_threshold, filter_method, nfeatures):
    if filepath == "":
        df = aux.load_dataset_dialog()
    else:
        df = aux.load_dataset_filepath(filepath)
    df = aux.removeLowFrequencyFeatures(df, unusable_feature_threshold)  # Apply threshold filter to full dataset (reduce runtime)
    runClassificationExperiment(df, BRF_undersampling, unusable_feature_threshold, filter_method, nfeatures)


if __name__ == "__main__":
    # Test parameters
    filename = 'C Elegans datasets/Version-1 datasets (no score threshold)/CElegans GOTerms dataset_v1.tsv' # set as empty ("") to have a file selection dialog
    #filename = ""  # set as empty ("") to have a file selection dialog
    unusable_feature_threshold = 10  # features with too few different values are removed in a preprocessing step
    BRF_undersampling = True  # True: undersample the training set to a 1:1 ratio (uses BRF for RF classifiers).
    filter_method = "AutoFilter"
    nfeatures = 0
    main(filename, BRF_undersampling, unusable_feature_threshold, filter_method, nfeatures)  # runs main function

