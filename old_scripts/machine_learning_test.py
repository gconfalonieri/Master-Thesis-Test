

import numpy as np
import pandas as pd
import toml
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

import models.utilities
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn import metrics

config = toml.load('config.toml')


def get_features_array(general_array):

    df_features = pd.DataFrame(columns=[])

    fpogx_mean = []
    fpogx_std = []
    fpogy_mean = []
    fpogy_std = []
    rpd_mean = []
    rpd_std = []
    lpd_mean = []
    lpd_std = []

    for x in general_array:
        fpogx_mean.append(np.mean(x[0]))
        fpogx_std.append(np.std(x[0]))
        fpogy_mean.append(np.mean([1]))
        fpogy_std.append(np.std(x[1]))
        rpd_mean.append(np.mean(x[2]))
        rpd_std.append(np.std(x[2]))
        lpd_mean.append(np.mean(x[3]))
        lpd_std.append(np.std(x[3]))

    df_features['FPOGX_MEAN'] = np.asarray(fpogx_mean).astype('float32')
    df_features['FPOGY_MEAN'] = np.asarray(fpogy_mean).astype('float32')
    df_features['RPD_MEAN'] = np.asarray(rpd_mean).astype('float32')
    df_features['LPD_MEAN'] = np.asarray(lpd_mean).astype('float32')
    df_features['FPOGX_STD'] = np.asarray(fpogx_std).astype('float32')
    df_features['FPOGY_STD'] = np.asarray(fpogy_std).astype('float32')
    df_features['RPD_STD'] = np.asarray(rpd_std).astype('float32')
    df_features['LPD_STD'] = np.asarray(lpd_std).astype('float32')

    return df_features


def get_labels_array():
    label_list = []
    path_labelled_df = config['path']['labelled_dataset']
    df_labelled = pd.read_csv(path_labelled_df)

    for i in df_labelled.index:
        user_id = df_labelled['USER_ID'][i]
        id = int((user_id.split('_'))[1])
        if id not in config['general']['excluded_users']:
            label_list.append(df_labelled['LABEL'][i])

    return np.asarray(label_list).astype('int')


general_array = models.utilities.get_questions_array()
complete_array = get_features_array(general_array)
labels_array = get_labels_array()

X_train, X_test, y_train, y_test = train_test_split(complete_array, labels_array, test_size=0.3, shuffle=True)

# clf = make_pipeline(StandardScaler(), svm.SVC(gamma='auto'))
clf = svm.SVC(kernel='linear', gamma='auto', C=2) # Linear Kernel
# clf = RandomForestClassifier(n_estimators=100, max_features='sqrt', max_depth=10, random_state=18)
# clf = KMeans( n_clusters=2, init='random', n_init=10, max_iter=10000,  tol=1e-02, random_state=0)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred))
print("Recall:", metrics.recall_score(y_test, y_pred))
