import tkinter as tk
from tkinter import filedialog
import pandas as pd


# Auxiliary function that populates a dict object with all candidate combinations of filter strategt and nfeatures
# Sets their initial score and variance values, to be filled during the strategy's execution in the internal CV
def create_candidate_filters_dict(nfeatures_array):
    return {("InfoGain", nfeatures_array[0]): {"score": 0, "variance": 0},
            ("InfoGain", nfeatures_array[1]): {"score": 0, "variance": 0},
            ("InfoGain", nfeatures_array[2]): {"score": 0, "variance": 0},
            ("InfoGain", nfeatures_array[3]): {"score": 0, "variance": 0},
            ("Chi2", nfeatures_array[0]): {"score": 0, "variance": 0},
            ("Chi2", nfeatures_array[1]): {"score": 0, "variance": 0},
            ("Chi2", nfeatures_array[2]): {"score": 0, "variance": 0},
            ("Chi2", nfeatures_array[3]): {"score": 0, "variance": 0},
            ("DecisionStump", nfeatures_array[0]): {"score": 0, "variance": 0},
            ("DecisionStump", nfeatures_array[1]): {"score": 0, "variance": 0},
            ("DecisionStump", nfeatures_array[2]): {"score": 0, "variance": 0},
            ("DecisionStump", nfeatures_array[3]): {"score": 0, "variance": 0},
            ("LogOddsRatio", nfeatures_array[0]): {"score": 0, "variance": 0},
            ("LogOddsRatio", nfeatures_array[1]): {"score": 0, "variance": 0},
            ("LogOddsRatio", nfeatures_array[2]): {"score": 0, "variance": 0},
            ("LogOddsRatio", nfeatures_array[3]): {"score": 0, "variance": 0},
            ("AsymmetricOptimalPrediction", nfeatures_array[0]): {"score": 0, "variance": 0},
            ("AsymmetricOptimalPrediction", nfeatures_array[1]): {"score": 0, "variance": 0},
            ("AsymmetricOptimalPrediction", nfeatures_array[2]): {"score": 0, "variance": 0},
            ("AsymmetricOptimalPrediction", nfeatures_array[3]): {"score": 0, "variance": 0}}


# creates an array with the candidate k values to be tried (this implementation of the AutoFilter is hard-coded for
# 4 values, this could be changed with some adjustments to the internalCV function, e.g., adding new elements to the
# dict with candidate methods_nfeature combinations
def set_nfeatures_array(feature_count):
    if feature_count > 1000:
        return [250, 500, 750, 1000]
    elif feature_count > 500:
        return [100, 200, 300, 400]
    else:
        return [25, 50, 75, 100]


def load_dataset_dialog():
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename()
    df = pd.read_csv(path, na_values='?', sep='\t', index_col=0)
    if 'STITCH_Code' in df:
        df = df.drop('STITCH_Code', 1)  # leave it in if we don't have stitch code
    if 'STITCH Code' in df:
        df = df.drop('STITCH Code', 1)  # leave it in if we don't have stitch code
    if 'STITCH_Compound' in df:
        df = df.drop('STITCH_Compound', 1)
    if 'STITCH Compound' in df:
        df = df.drop('STITCH Compound', 1)
    if 'InteractorsList' in df:
        df = df.drop('InteractorsList', 1)
    if 'InteractorsCount' in df:
            df = df.drop('InteractorsCount', 1)
    return df


def load_dataset_filepath(filepath):
    df = pd.read_csv(filepath, na_values='?', sep='\t', index_col=0)
    if 'STITCH_Code' in df:
        df = df.drop('STITCH_Code', 1)  # leave it in if we don't have stitch code
    if 'STITCH Code' in df:
        df = df.drop('STITCH Code', 1)  # leave it in if we don't have stitch code
    if 'STITCH_Compound' in df:
        df = df.drop('STITCH_Compound', 1)
    if 'STITCH Compound' in df:
        df = df.drop('STITCH Compound', 1)
    if 'InteractorsList' in df:
        df = df.drop('InteractorsList', 1)
    if 'InteractorsCount' in df:
            df = df.drop('InteractorsCount', 1)
    return df



def printdict(candidate_filters):
    for fmethod, nfeatures in candidate_filters:
        print(f'Candidate Filter: {fmethod}, k: {nfeatures}. Median AUC Score: {round(candidate_filters[fmethod, nfeatures]["score"], 3)}, '
              f'AUC variance: {round(candidate_filters[fmethod, nfeatures]["variance"], 3)}')


# remove unusable features based on a threshold that is the minimum number of 'non-zero' values in a feature
# works even if the feature has a different 'majority value' than 0, as we get the count of the most common value
def removeLowFrequencyFeatures(df, threshold):
    if threshold < 0:
        return df
    number_instances = df.shape[0]
    for feature in df:
        try:
            mostFrequent = df[feature].value_counts(ascending=True)[0]
        except:
            mostFrequent = 0
        if number_instances - mostFrequent < threshold:
            df = df.drop(feature, 1)
    return df


# remove unusable features based on a threshold that is the minimum number of 'non-zero' values in a feature
# works even if the feature has a different 'majority value' than 0, as we get the count of the most common value
def removeLowFrequencyFeatures_TrainTest(training_set, test_set, threshold):
    if threshold < 0:
        return training_set, test_set
    number_instances = training_set.shape[0]
    for feature in training_set:
        try:
            mostFrequent = training_set[feature].value_counts(ascending=True)[0]
        except:
            mostFrequent = 0
        if number_instances - mostFrequent < threshold:
            training_set = training_set.drop(feature, 1)
            test_set = test_set.drop(feature, 1)
    return training_set, test_set



# Receives a dict object associating with candidate filter and nfeatures (k value) pair with a score and its variance
# Compares the scores and returns the highest-score combination, using the lowest variance as a tie-breaking criterion
def select_autofilter_strategy(candidate_filters):
    chosen_filter = "default"
    chosen_nfeatures = 0
    highest_score = 0
    lowest_variance = 1
    for fmethod, nfeatures in candidate_filters:
        if candidate_filters[fmethod, nfeatures]["score"] > highest_score:
            highest_score = candidate_filters[fmethod, nfeatures]["score"]
            chosen_filter = fmethod
            chosen_nfeatures = nfeatures
            lowest_variance = candidate_filters[fmethod, nfeatures]["variance"]
        elif candidate_filters[fmethod, nfeatures]["score"] == highest_score:
            if candidate_filters[fmethod, nfeatures]["variance"] < lowest_variance:
                highest_score = candidate_filters[fmethod, nfeatures]["score"]
                chosen_filter = fmethod
                chosen_nfeatures = nfeatures
                lowest_variance = candidate_filters[fmethod, nfeatures]["variance"]
    if chosen_filter == "default":
        print('Error: No candidate winner method found!')
        print("Exiting the program...")
        exit(0)
    print(f"Chosen strategy: {chosen_filter}_{chosen_nfeatures}. Average score {highest_score} and variance {lowest_variance}")
    return chosen_filter, chosen_nfeatures

