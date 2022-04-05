import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

DOWNLOAD_ROOT = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/"
RED_URL = DOWNLOAD_ROOT + "winequality-red.csv"
WHITE_URL = DOWNLOAD_ROOT + "winequality-white.csv"
DATASETS_PATH = "./datasets"

def fetch_wine_data(red_url=RED_URL, white_url=WHITE_URL, datasets_path=DATASETS_PATH):
    if not os.path.isdir(datasets_path):
        os.mkdir(datasets_path)
    red_wine_path = os.path.join(datasets_path, "red_wine.csv")
    urllib.request.urlretrieve(red_url, red_wine_path)
    white_wine_path = os.path.join(datasets_path, "white_wine.csv")
    urllib.request.urlretrieve(white_url, white_wine_path)

def load_red_wine_data(datasets_path=DATASETS_PATH):
    red_csv_path = os.path.join(datasets_path, "red_wine.csv")
    return pd.read_csv(red_csv_path, sep=";")

def load_white_wine_data(datasets_path=DATASETS_PATH):
    white_csv_path = os.path.join(datasets_path, "white_wine.csv")
    return pd.read_csv(white_csv_path, sep=";")

def add_color_feature(red_df, white_df):
    red_df["color"] = 1
    white_df["color"] = 0

def concat_dataframes(df1, df2):
    return pd.concat([df1, df2], ignore_index=True)

def split_dataset(data, test_size=0.2, random_state=42):

    # randomly subdivide dataset, using stratified sampling to maintain color category 
    # proportions of the full dataset
    split = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    for train_index, test_index in split.split(data, data["color"]):
        strat_train_set = data.loc[train_index]
        strat_test_set = data.loc[test_index]

    #split the color labels off of both datasets
    train_labels = strat_train_set["color"].copy()
    strat_train_set = strat_train_set.drop("color", axis=1)
    test_labels = strat_test_set["color"].copy()
    strat_test_set = strat_test_set.drop("color", axis=1)

    #return split datasets and labels
    return strat_train_set, train_labels, strat_test_set, test_labels