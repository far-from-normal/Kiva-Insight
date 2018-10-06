import pandas as pd
import numpy as np
from data_params import Data
from pathlib import Path
import itertools
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib
matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
import seaborn as sns


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



def print_confusion_matrix(confusion_matrix, class_names, figsize = (8, 5.5), fontsize=14):
    """Prints a confusion matrix, as returned by sklearn.metrics.confusion_matrix, as a heatmap.

    Arguments
    ---------
    confusion_matrix: numpy.ndarray
        The numpy.ndarray object returned from a call to sklearn.metrics.confusion_matrix.
        Similarly constructed ndarrays can also be used.
    class_names: list
        An ordered list of class names, in the order they index the given confusion matrix.
    figsize: tuple
        A 2-long tuple, the first value determining the horizontal size of the ouputted figure,
        the second determining the vertical size. Defaults to (10,7).
    fontsize: int
        Font size for axes labels. Defaults to 14.

    Returns
    -------
    matplotlib.figure.Figure
        The resulting confusion matrix figure
    """

    # fig_dpi = 400
    # fig_size = (8, 5.5)
    # plt.figure(figsize=fig_size)
    #
    # ax = sns.heatmap(result_precision, annot=True, fmt=".4f", cmap="Greens_r", square=True)
    # ax.set(xlabel="L2 / ridge = 0             Penalty                  L1 / lasso = 1")
    # ax.set(ylabel="Regualization (alpha = 1/C)")
    # fig = ax.get_figure()
    # fig.savefig("conf_matrix.png", dpi=fig_dpi)
    # plt.close()


    df_cm = pd.DataFrame(
        confusion_matrix, index=class_names, columns=class_names,
    )
    fig = plt.figure(figsize=figsize)
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['mathtext.it'] = 'Arial:italic'
    plt.rcParams['mathtext.bf'] = 'Arial:bold'
    plt.rcParams['mathtext.rm'] = 'Arial'
    try:
        heatmap = sns.heatmap(df_cm, annot=True, fmt="d", cmap="Greens", square=True, annot_kws={"size": 20})
    except ValueError:
        raise ValueError("Confusion matrix values must be integers.")
    heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=90, ha='right', fontsize=fontsize)
    heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, fontsize=fontsize)
    # plt.ylabel('True outcome', fontsize=16)
    # plt.xlabel('Predicted outcome', fontsize=16)
    plt.xlabel(r"$\mathbf{Predicted\ outcome}$", fontsize=16)
    plt.ylabel(r"$\mathbf{True\ outcome}$", fontsize=16)
    fig = heatmap.get_figure()
    fig.savefig("conf_matrix_.png", dpi=400)
    plt.close()
    return None



data_par = Data()
dir_to_saved_data = data_par.dir_to_saved_data


path_to_scores = Path(dir_to_saved_data, "cv_results_.csv")
df_scores = pd.read_csv(path_to_scores)
df_scores = df_scores[["param_clf__alpha", "param_clf__l1_ratio", "mean_test_accuracy", "std_test_accuracy", "mean_test_precision", "std_test_precision"]]
df_scores
result_precision = df_scores.copy()
result_accuracy = df_scores.copy()
result_precision = result_precision.pivot(index='param_clf__alpha', columns='param_clf__l1_ratio', values='mean_test_precision')
result_accuracy = result_accuracy.pivot(index='param_clf__alpha', columns='param_clf__l1_ratio', values='mean_test_accuracy')


path_to_responses = Path(dir_to_saved_data, "y_test_y_pred.csv")
df_responses = pd.read_csv(path_to_responses)
print(df_responses.head())
classes = ["not funded", "funded"]
conf_mat = confusion_matrix(df_responses["test"], df_responses["pred"])
# cmap = sns.diverging_palette(150, 275, s=80, l=55, as_cmap=True)
# cmap = sns.diverging_palette(150, 275, s=80, l=55, as_cmap=True)

fig_dpi = 400
fig_size = (8, 5.5)
plt.figure(figsize=fig_size)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Arial:italic'
plt.rcParams['mathtext.bf'] = 'Arial:bold'
plt.rcParams['mathtext.rm'] = 'Arial'
ax = sns.heatmap(result_precision, annot=True, fmt=".4f", cmap="Greens_r", square=True)
# ax.set(xlabel="L2 / ridge = 0             Penalty                  L1 / lasso = 1")
# ax.set(ylabel="Regualization (alpha = 1/C)")
plt.xlabel("L2 / ridge = 0             " + r"$\mathbf{Penalty}$" + "                  L1 / lasso = 1", fontsize=16)
plt.ylabel(r"$\mathbf{Regularization}$" + " (" + r"$\alpha = 1/C$" + ")", fontsize=16)
fig = ax.get_figure()
fig.savefig("hyperparameter_heatmap_precision.png", dpi=fig_dpi)
plt.close()


print_confusion_matrix(conf_mat, classes, figsize=(8,5.5), fontsize=14)







#
#
# df_scores = pd.concat(df_scores_list)
# df_scores.rename(columns={'Unnamed: 0': 'train-test'}, inplace=True)
# df_scores.reset_index(inplace=True)
# df_scores.drop(columns=["index"], inplace=True)
# df_reg_coef = pd.concat(df_reg_coef_list)
# df_reg_coef.rename(columns={'Unnamed: 0':' train-test'}, inplace=True)
# df_reg_coef.reset_index(inplace=True)
# df_reg_coef.drop(columns=["index"], inplace=True)
# df_reg_coef["Percentage of non-zero coefficients"] = 100.0 - df_reg_coef["Ratio of zero"]
# df_reg_coef["group"] = "1"
# max_features = df_reg_coef["Number of coefficients"][0]
# df_reg_coef
# df_scores_precision = df_scores[["L1 ratio", "train-test", "precision"]]
# df_scores_precision = df_scores_precision[~df_scores_precision["train-test"].isin(["train"])]
# df_scores_precision
# # %%
# sns.set_style("ticks")
#
#
# fig_dpi = 400
# fig_size = (5, 3)
# plt.figure(figsize=fig_size)
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = 'Arial'
# ax = sns.lineplot(x="L1 ratio", y="precision",
#     hue="train-test", style="train-test", alpha=0.75,
#     markers=True, dashes=False, data=df_scores_precision)
#
#
# ax.figure.tight_layout()
# plt.legend(
#     title="",
#     # bbox_to_anchor=legend_coods_tags,
#     frameon=False,)
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles=handles[1:], labels=labels[1:])
# ax.set(xlabel="L2 / ridge = 0                                           L1 / lasso = 1")
# ax.set(ylabel="Precision")
# sns.despine(offset = 10)
# # ax.set(ylim=(0.97, 0.985))
# # ax.set(ylim=(0.9, 1))
#
# fig = ax.get_figure()
# fig.savefig("precision_vs_regularization.png", dpi=fig_dpi)
# plt.close()
#
#
# # %%
# fig_dpi = 400
# fig_size = (5, 3)
# plt.figure(figsize=fig_size)
# plt.rcParams['font.family'] = 'sans-serif'
# plt.rcParams['font.sans-serif'] = 'Arial'
# ax = sns.lineplot(x="L1 ratio", y="Percentage of non-zero coefficients",
#     style="group", markers=True, data=df_reg_coef, legend=False)
#
#
# ax.figure.tight_layout()
# ax.set(xlabel="L2 / ridge = 0                                           L1 / lasso = 1")
# ax.set(ylabel="Percentage of non-zero coefficients")
# ax.set(ylim=(0, 100))
# # ax.set(xlim=(0, 1))
# ax.text(0.025, 95, "113,000 features", horizontalalignment='left', size='medium', color='black', weight='semibold')
# ax.text(0.75, 25, "25,000 features", horizontalalignment='left', size='medium', color='black', weight='semibold')
# sns.despine(offset = 10)
# fig = ax.get_figure()
# fig.savefig("percentage_of_coefs_equal_to_0_vs_regularization.png", dpi=fig_dpi)
# plt.close()
