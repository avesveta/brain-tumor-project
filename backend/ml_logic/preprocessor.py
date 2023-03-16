import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ml_logic.parameters import *
import os
import nibabel as nib

def preprocess_split_model_1(pkl_path,channel):
    """
    get the channel name as input, load the pkl file from raw_data folder and then split the data.
    scale, balance the train data.
    output is the X_train, X_test, y_train, y_test
    """

    # load data

    df = pd.read_pickle(pkl_path) ##hier I need to ask question.
    #encode the target columns
    df['Grade'] = df['Grade'].apply(lambda x: 1 if x == 'HGG' else 0)

    # crop images
    df[f'{channel}_nii'] = df[f'{channel}_nii'].apply(lambda x: np.array(x[MIN_HEIGHT:MAX_HEIGHT,MIN_WIDTH:MAX_WIDTH,MIN_DEPTH:MAX_DEPTH]))
    X = df[f'{channel}_nii']
    y = df['Grade']

    # reshape input data
    X = np.array([np.array(val) for val in X])

    #use min-max Scaler to scale the data. 840 is the median of max values of each image
    if channel != 'seg':
        X[X>840]=840 # set the max value to 840
        X = (X - 0)/840 # Min-Max Scaler

    #reshape the X to fit the input of Model
    X = X.reshape(len(X), X[0].shape[0], X[0].shape[1], X[0].shape[2], 1)

    # the target columns have to be updated, it's only the fremework
    # create train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=SEED)
    # reset the index of y, so that the indexes of X and y match
    y_train = y_train.reset_index(drop=True)
    y_test = y_test.reset_index(drop=True)

    #need to add extra LGGs to the train data so it is balanced
    #if y_train.value_counts()[0] != y_train.value_counts()[1]:

    # find indices of positive examples
    pos_indices = np.where(y_train == 0)[0]

    # perform data augmentation on positive examples
    rotated_examples1 = np.rot90(X_train[pos_indices], axes=(1, 2))
    rotated_examples2 = np.rot90(rotated_examples1, axes=(1, 2))

    # append new images and labels to the training set
    X_train = np.concatenate([X_train, rotated_examples1, rotated_examples2], axis=0)
    y_train = np.concatenate([y_train, np.repeat(y_train[pos_indices], 2)], axis=0)

    return X_train, X_test, y_train, y_test

def preprocess_features_model_2(X: pd.DataFrame) -> np.ndarray:

    def survival_encoding(x):
        if x <= 180.00:
            return 0
        elif x <= 515.00:
            return 1
        else:
            return 2

    # the target columns have to be updated, it's only the fremework
    X['Survival_days'] = X['Survival_days'].apply(survival_encoding)

    X_train, X_test, y_train, y_test = train_test_split(X[['Age', 'Grade']], \
        X['Survival_days'], test_size=0.2, random_state=42)

def preprocess_nii_for_test(nii_file):
    img = nib.load(nii_file)
    img_data = img.dataobj
    img_data_choped = img_data[MIN_HEIGHT:MAX_HEIGHT,
                               MIN_WIDTH:MAX_WIDTH,
                               MIN_DEPTH:MAX_DEPTH]
    img_data_choped = np.array(img_data_choped) # not sure if that works
    img_data_choped[img_data_choped>840]=840 # set the max value to 840
    img_data_choped = (img_data_choped - 0)/840 # Min-Max Scaler
