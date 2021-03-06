import pandas as pd
import numpy as np
import itertools
from time import time
from pathlib import Path
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
# from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from data_utils import preprocess_train_df, save_coefs, create_pipeline
from data_params import Data
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")


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

    # return plt
    np.set_printoptions(precision=2)
    fig.savefig(name, dpi=300, bbox_inches="tight")

    return None


# %%
# classifier pipelines
pipeline_type = "enet"  # "lr" "enet" "rf"

# scoring metrics to report
scoring = {
    "accuracy": "accuracy",
    # "balanced_accuracy": "balanced_accuracy",
    # "average_precision": "average_precision",
    # "f1": "f1",
    # "f1_micro": "f1_micro",
    # "f1_macro": "f1_macro",
    # "f1_weighted": "f1_weighted",
    # 'f1_samples': 'f1_samples',
    "precision": "precision",
    # "precision_weighted": "precision_weighted",
    # "recall": "recall",
    # "roc_auc": "roc_auc",
}

# hyper-parameter search grid
param_grid = [{'clf__alpha': [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7],
        'clf__l1_ratio': [0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0]}]

# %% ############################

data_par = Data()
col_label = data_par.cols_label
cols_process = data_par.cols_process
cols_output = data_par.cols_output
valid_status = data_par.valid_status
dir_to_saved_data = data_par.dir_to_saved_data
# dir_to_saved_data = "saved_data_1"
dir_to_query_data = data_par.dir_to_query_data
path_to_training_data = data_par.path_to_training_data

# n_jobs is high for AWS EC@ m5d.12xlarge Instance
predict = False
num_cv = 5
n_jobs = 20
verbose = 20
refit = "precision"
downsample_rate = 1

# load data
df = pd.read_csv(
    path_to_training_data, usecols=cols_process, skiprows=lambda i: i % downsample_rate != 0
)
# select and transform columns
df = preprocess_train_df(df, valid_status, cols_output, predict=False)
# y and X column names
cols_y = col_label
cols_X = [x for x in cols_output if x != cols_y]
# transform label
lb = LabelBinarizer()
labels = lb.fit_transform(df[cols_y]).ravel()
classes = lb.classes_
classes = ["not funded", "funded"]
df.drop(columns=[cols_y], inplace=True)
# create train, test data
X_train, X_test, y_train, y_test = train_test_split(df, labels, test_size=0.2, random_state=42)

start = time()

# %%
# set pipeline
pipeline = create_pipeline(pipeline_type)
# set grid search
grid_search = GridSearchCV(
    pipeline,
    param_grid=param_grid,
    cv=num_cv,
    scoring=scoring,
    n_jobs=n_jobs,
    refit=refit,
    )
# cross validate gris search
grid_search.fit(X_train, y_train)

# %%
print("Testing Model")
# evaluate the 20% hold-out test data
y_pred = grid_search.best_estimator_.predict(X_test)
precision = precision_score(y_test, y_pred, average='binary')
accuracy = accuracy_score(y_test, y_pred)

# output true and predicted labels
df_test_pred = pd.DataFrame({"test": y_test, "pred": y_pred})
df_test_pred.to_csv(Path(dir_to_saved_data, "y_test_y_pred.csv"), index=False)

# plot raw and normalized confusion matrices
conf_mat = confusion_matrix(y_test, y_pred)
cm_plot_filename_norm = Path(dir_to_saved_data, pipeline_type + "_norm_confusion.png")
cm_plot_filename_raw = Path(dir_to_saved_data, pipeline_type + "_raw_confusion.png")
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

# output cv scores
df_results = pd.DataFrame(grid_search.cv_results_)
df_results.to_csv(Path(dir_to_saved_data, "cv_results_.csv"), index=False)

# output params from best pipeline
df_best_params = pd.DataFrame(grid_search.best_params_, index=[0])
df_best_params.to_csv(Path(dir_to_saved_data, "best_params_.csv"), index=False)

# output test results
df_test_results = pd.DataFrame({"accuracy": accuracy, "precision": precision}, index=[0])
df_test_results.to_csv(Path(dir_to_saved_data, "test_results.csv"), index=False)

# output best saved pipeline
path_to_saved_model = Path(dir_to_saved_data, pipeline_type + "_model.pkl")
joblib.dump(grid_search.best_estimator_, path_to_saved_model)

# output coefficients from best pipeline
save_coefs(grid_search.best_estimator_, dir_to_saved_data, pipeline_type)

stop = time()
print(stop - start)
