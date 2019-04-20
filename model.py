import os
import glob
from time import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


def train_classifier(clf, x_train, y_train):
    ''' Fits a classifier to the training data. '''

    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(x_train, y_train)
    end = time()

    # Print the results
    print(f"Trained model in {round(end - start, 3)} seconds")


def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''

    y_pred = clf.predict(features)
    return f1_score(target, y_pred, pos_label='H'), sum(target == y_pred) / float(len(y_pred))


def train_predict(clf, x_train, y_train, x_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''

    # Indicate the classifier and the training set size
    print(f"Training a {clf.__class__.__name__} using a training set size of {len(x_train)}. . .")

    # Train the classifier
    train_classifier(clf, x_train, y_train)

    # Print the results of prediction for both training and testing
    f1, acc = predict_labels(clf, x_train, y_train)
    print(f1, acc)
    print(f"F1 score and accuracy score for training set: {f1} , {np.mean(acc)}.")

    f1, acc = predict_labels(clf, x_test, y_test)
    print(f"F1 score and accuracy score for test set: {f1} , {np.mean(acc)}.")


SEASON_TO_TEST = 22018  # SEASON_ID for specified test data
LATEST_DATA = True  # FOLDER for specified feature data

root = os.path.join(os.getcwd(), 'feature_data')
if LATEST_DATA:
    sub_folder = max(glob.iglob(os.path.join(root, '*/')), key=os.path.getmtime)
else:
    folder_ = ''
    if not folder_:
        raise Exception("Source file name must be specified, assign to 'source_file' var")
    sub_folder = os.path.join(root, folder_)

features = np.load(os.path.join(sub_folder, 'x_train.npy'))
labels = np.load(os.path.join(sub_folder, 'y_train.npy'))
season_lbl = np.load(os.path.join(sub_folder, 'season_lbl.npy'))

for SEASON_TO_TEST in set(season_lbl):
    if SEASON_TO_TEST:
        train_indices = [i for i, v in enumerate(season_lbl) if int(v) != SEASON_TO_TEST]
        test_indices = [i for i, v in enumerate(season_lbl) if int(v) == SEASON_TO_TEST]

        x_train, y_train = features[train_indices], labels[train_indices]
        x_test, y_test = features[test_indices], labels[test_indices]
    else:
        x_train, x_test, y_train, y_test = \
            train_test_split(features,
                             labels,
                             test_size=0.2,
                             random_state=2,
                             stratify=labels)

    clf_A = LogisticRegression(random_state=42, solver='saga', max_iter=999999)
    train_predict(clf_A, x_train, y_train, x_test, y_test)
    print('')

# clf_B = SVC(random_state=912, kernel='rbf', gamma='auto')
# train_predict(clf_B, x_train, y_train, x_test, y_test)
# print('')
#
# clf_C = KNeighborsClassifier(n_neighbors=8)
# train_predict(clf_C, x_train, y_train, x_test, y_test)
# print('')
#
# clf_D = LinearDiscriminantAnalysis()
# train_predict(clf_D, x_train, y_train, x_test, y_test)
# print('')
#
# clf_E = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
# train_predict(clf_E, x_train, y_train, x_test, y_test)
# print('')
