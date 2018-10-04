from pathlib import Path
import pandas as pd
import numpy as np
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid")

from sklearn.externals import joblib

from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split

from imblearn.over_sampling import SMOTE

from sklearn.metrics import classification_report
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import confusion_matrix

from classifier_train_params import ClfDef


def classification_report_csv(report, filename):
    report_data = []
    lines = report.split("\n")
    for line in lines[2:-3]:
        row = {}
        row_data = line.split("      ")
        row["class"] = row_data[0]
        row["precision"] = float(row_data[1])
        row["recall"] = float(row_data[2])
        row["f1_score"] = float(row_data[3])
        row["support"] = float(row_data[4])
        report_data.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe.to_csv(filename, index=False)
    return None


def plot_confusion_matrix(
    cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
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

    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")


# %% ####################################


def train(
    verbose,
    n,
    n_jobs,
    num_cv,
    num_iter,
    resampling,
    clf_type,
    input_data_filename,
    saved_data_dir,
):

    scoring_metric = {"MCC": make_scorer(matthews_corrcoef)}
    # scoring_metric = "recall" #"neg_log_loss" # "roc_auc" "precision" "recall" "f1"

    clf = ClfDef(clf_type)
    trained_base_name = clf.trained_base_name
    param_dist = clf.param_dist
    pipeline = clf.pipeline
    clf_family = clf.family

    input_data_filename = "loans_Transformed.csv"
    saved_data_dir = Path(saved_data_dir)
    if saved_data_dir.is_dir() is False:
        saved_data_dir.mkdir(parents=True, exist_ok=True)
    output_best_score_filename = saved_data_dir.joinpath(
        clf_type + "_" + str(n) + "_score_MCC.txt"
    )
    output_best_params_filename = saved_data_dir.joinpath(
        clf_type + "_" + str(n) + "_params.txt"
    )
    output_best_report_filename = saved_data_dir.joinpath(
        clf_type + "_" + str(n) + "_report.csv"
    )
    output_best_coef_filename = saved_data_dir.joinpath(
        clf_type + "_" + str(n) + "_coefs_feat_imp.csv"
    )
    output_best_cm_filename = saved_data_dir.joinpath(
        clf_type + "_" + str(n) + "_cm.csv"
    )
    output_best_trained_model_filename = saved_data_dir.joinpath(
        trained_base_name + "_" + str(n) + ".pkl"
    )
    output_best_cm_plot_filename = saved_data_dir.joinpath(
        clf_type + "_" + str(n) + "_cm.png"
    )

    # %%

    max_num_iter = np.prod(np.array([len(y) for y in list(param_dist.values())]))
    n_iter = min(num_iter, max_num_iter)

    df = pd.read_csv(input_data_filename, skiprows=lambda i: i % n != 0)
    df.drop(columns=["LOAN_ID"], inplace=True)
    label_col = "STATUS"

    feats_col = df.columns.difference([label_col])
    df[label_col] = pd.Categorical(df[label_col], ordered=True)

    y, label_col_vals = pd.factorize(df[label_col], sort=True)
    X = df[feats_col]

    print("################## VALUES, LABELS: ", y, label_col_vals)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    if resampling == True:
        sm = SMOTE(random_state=0)
        X_train, y_train = sm.fit_sample(X_train, y_train.ravel())
        print("After OverSampling, the shape of train_X: {}".format(X_train.shape))
        print("After OverSampling, the shape of train_y: {} \n".format(y_train.shape))

    search = RandomizedSearchCV(
        pipeline,
        param_distributions=param_dist,
        cv=num_cv,
        n_iter=n_iter,
        scoring=scoring_metric,
        n_jobs=n_jobs,
        verbose=verbose,
        refit="MCC",
    )

    search.fit(X_train, y_train)
    joblib.dump(search.best_estimator_, output_best_trained_model_filename)

    predictions = search.predict(X_test)

    report = classification_report(y_test, predictions)
    print(report)
    print("MCC: ", matthews_corrcoef(y_test, predictions))

    # conf_matrix
    preds = label_col_vals[search.predict(X_test[feats_col])]
    conf_mat_pd = pd.crosstab(
        label_col_vals[y_test], preds, rownames=["Actual"], colnames=["Predicted"]
    )

    # %% Outputting reults to file

    if clf_family is "linear":
        coef = pd.DataFrame(
            -search.best_estimator_.steps[1][1].coef_[0],
            index=X.columns,
            columns=["coefficient"],
        ).sort_values("coefficient", ascending=False)
        coef.to_csv(output_best_report_filename, sep=",")
        coef.reset_index(level=0, inplace=True)
        coef.columns = ["FeatureName", "Value"]
        # write coefs
        coef.to_csv(output_best_coef_filename, sep=",", index=False)
        print(coef)
    else:
        feature_importances = pd.DataFrame(
            search.best_estimator_.steps[1][1].feature_importances_,
            index=X.columns,
            columns=["importance"],
        ).sort_values("importance", ascending=False)
        feature_importances.reset_index(level=0, inplace=True)
        feature_importances.columns = ["FeatureName", "Value"]
        # write feature_importances
        feature_importances.to_csv(output_best_coef_filename, sep=",", index=False)
        print(feature_importances)

    # write score MCC
    with open(output_best_score_filename, "w") as the_file:
        the_file.write("Best score: %0.3f\n" % search.best_score_)
        the_file.write("MCC score: %0.3f\n" % matthews_corrcoef(y_test, predictions))

    # write best params
    print("Best parameters set:")
    best_parameters = search.best_estimator_.get_params()
    with open(output_best_params_filename, "w") as the_file:
        for param_name in sorted(param_dist.keys()):
            the_file.write("%s:\t %r\n" % (param_name, best_parameters[param_name]))

    # write classification report
    classification_report_csv(report, output_best_report_filename)

    # write confusion matrix filename
    print(conf_mat_pd)
    conf_mat_pd.to_csv(output_best_cm_filename, sep=",")

    cm = confusion_matrix(y_test, search.predict(X_test[feats_col]))
    np.set_printoptions(precision=2)
    fig = plt.figure()
    # plot_confusion_matrix(cnf_matrix, classes=label_col_vals, normalize=True,
    #                       title='Normalized confusion matrix')

    normalize = True
    title = "Confusion matrix"
    cmap = plt.cm.Blues
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
    fig = plt.figure(figsize=(4, 4))
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(label_col_vals))
    plt.xticks(tick_marks, label_col_vals, rotation=45)
    plt.yticks(tick_marks, label_col_vals)

    fmt = ".2f" if normalize else "d"
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            format(cm[i, j], fmt),
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )

    plt.tight_layout()
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    # plt.show()
    # fig = plt.figure(figsize=(4,4))
    fig.savefig(output_best_cm_plot_filename, dpi=300, bbox_inches="tight")
