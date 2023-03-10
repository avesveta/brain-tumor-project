
import numpy as np
import pandas as pd
from nibabel.viewers import OrthoSlicer3D
from nibabel import nifti1
import nibabel as nib
from matplotlib import pylab as plt

from ml_logic.data import load_nii_from_gcp

from google.cloud import storage
import os

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import resample

def preprocess_features_model_1(X: pd.DataFrame) -> np.ndarray:
    X['Grade'] = X['Grade'].apply(lambda x: 0 if x == 'HGG' else 1)

    # the target columns have to be updated, it's only the fremework
    X_train, X_test, y_train, y_test = train_test_split(X['nii'], X['Grade'], test_size=0.2, random_state=42)

    #need to add extra LGGs to the train data so it is balanced
    #if y_train.value_counts()[0] != y_train.value_counts()[1]:

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
