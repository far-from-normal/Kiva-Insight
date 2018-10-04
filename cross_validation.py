import pandas as pd
import numpy as np
import itertools
from time import time
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import seaborn as sns

from sklearn.externals import joblib

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn.model_selection import cross_val_predict
from sklearn import metrics

from data_utils import preprocess_train_df, save_coefs, create_pipeline
from data_params import Data


def plot_confusion_matrix(
    cm, classes, name, normalize=False, title="Confusion matrix", cmap=plt.cm.Greens
):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """

    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print("Confusion matrix, without normalization")

    print(cm)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    # plt.colorbar(labelsize=16)
    cb = plt.colorbar()
    cb.ax.tick_params(labelsize=16)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, fontsize=16)
    plt.yticks(tick_marks, classes, fontsize=16)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            fontsize=16,
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.title(title, fontsize=24)
    plt.ylabel("Actual Outcome", fontsize=20)
    plt.xlabel("Predicted Outcome", fontsize=20)

    # ax.set_title(title,fontsize= 30) # title of plot
    # ax.set_xlabel("Actual Outcome",fontsize = 20) #xlabel
    # ax.set_ylabel("Actual Outcome", fontsize = 20)#ylabel

    # return plt
    np.set_printoptions(precision=2)
    fig.savefig(name, dpi=300, bbox_inches="tight")

    return None


# %% classifier pipelines
pipeline_type = "enet"  # "lr" "enet" "rf"

scoring = {
    "accuracy": "accuracy",
    "balanced_accuracy": "balanced_accuracy",
    "average_precision": "average_precision",
    "f1": "f1",
    "f1_micro": "f1_micro",
    "f1_macro": "f1_macro",
    "f1_weighted": "f1_weighted",
    # 'f1_samples': 'f1_samples',
    "precision": "precision",
    "precision_weighted": "precision_weighted",
    "recall": "recall",
    "roc_auc": "roc_auc",
    "roc_auc": "roc_auc",
}

# %% ############################

data_par = Data()
l1_ratio = data_par.l1_ratio
cols_process = data_par.cols_process
cols_output = data_par.cols_output
valid_status = data_par.valid_status
dir_to_saved_data = data_par.dir_to_saved_data
dir_to_query_data = data_par.dir_to_query_data
path_to_training_data = data_par.path_to_training_data

predict = False
num_cv = 5
n_jobs = 5
verbose = 2
downsample_rate = 10

df = pd.read_csv(
    path_to_training_data, usecols=cols_process, skiprows=lambda i: i % downsample_rate != 0
)
df = preprocess_train_df(df, valid_status, cols_output, predict=False)
#
cols_y = "STATUS"
cols_X = [x for x in cols_output if x != cols_y]

lb = LabelBinarizer()
labels = lb.fit_transform(df[cols_y]).ravel()
classes = lb.classes_
classes = ["not funded", "funded"]
print(labels)
print(classes)
df.drop(columns=[cols_y], inplace=True)

X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=42)

for l1_rat in l1_ratio:

    start = time()

    print("L1_ratio: ", l1_rat)
    pipeline = create_pipeline(pipeline_type, l1_rat)

    # %%
    print("Cross-Validating Model")
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        scoring=scoring,
        cv=num_cv,
        return_train_score=True,
        verbose=verbose,
        n_jobs=n_jobs,
    )
    pipeline.fit(X_train, y_train)
    # print(scores)

    avg_scores = dict()
    for key, val in scores.items():
        avg_scores[key] = np.mean(scores[key])
    # print(avg_scores)

    avg_scores_train_test = {"test": {}, "train": {}}
    for key in avg_scores:
        if key.startswith("test_"):
            sub_s = str(key)
            sub_s = sub_s.replace("test_", "")
            avg_scores_train_test["test"][sub_s] = avg_scores[key]
        if key.startswith("train_"):
            sub_s = str(key)
            sub_s = sub_s.replace("train_", "")
            sub_s = key.replace("train_", "")
            avg_scores_train_test["train"][sub_s] = avg_scores[key]
    # print(avg_scores_train_test)

    # %%
    print("Testing Model")
    # y_pred = cross_val_predict(pipeline, df[cols_X], labels, cv=num_cv, verbose=verbose, n_jobs=n_jobs)
    y_pred = pipeline.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred, average='macro')
    precision = precision_score(y_test, y_pred, average='binary')
    precision_weighted = precision_score(y_test, y_pred, average='weighted')
    accuracy = accuracy_score(y_test, y_pred)
    print("############## Precision_weighted: ", precision_weighted)

    conf_mat = confusion_matrix(y_test, y_pred)
    cm_plot_filename_norm = Path(dir_to_saved_data, pipeline_type + "_" + str(l1_rat) + "_norm_confusion.png")
    cm_plot_filename_raw = Path(dir_to_saved_data, pipeline_type + "_" + str(l1_rat) + "_raw_confusion.png")
    plot_confusion_matrix(
        conf_mat,
        classes,
        cm_plot_filename_norm,
        normalize=True,
        title="Confusion matrix",
        cmap=plt.cm.Greens,
    )
    plot_confusion_matrix(
        conf_mat,
        classes,
        cm_plot_filename_raw,
        normalize=False,
        title="Confusion matrix",
        cmap=plt.cm.Greens,
    )

    print("Training Model")
    # pipeline.fit(df[cols_X], labels)
    path_to_saved_model = Path(dir_to_saved_data, pipeline_type + "_" + str(l1_rat) + "_model.pkl")
    joblib.dump(pipeline, path_to_saved_model)
    save_coefs(pipeline, dir_to_saved_data, pipeline_type, str(l1_rat) + "_")

    # create scores dataframe
    df_avg_score = pd.DataFrame.from_dict(avg_scores_train_test, orient="index")
    res = pd.DataFrame(columns=df_avg_score.columns)
    res = res.append([{
        "roc_auc": roc_auc,
        "precision": precision,
        "precision_weighted": precision_weighted,
        "accuracy": accuracy}], ignore_index=True)
    df_avg_score = df_avg_score.append(res)
    df_avg_score.rename({df_avg_score.index[0]: "validation"}, inplace=True)
    df_avg_score.rename({df_avg_score.index[1]: "train"}, inplace=True)
    df_avg_score.rename({df_avg_score.index[2]: "test"}, inplace=True)
    print(df_avg_score)
    df_avg_score.to_csv(Path(dir_to_saved_data, pipeline_type + "_" + str(l1_rat) + "_saved_score.csv"))

    stop = time()
    print(stop - start)
