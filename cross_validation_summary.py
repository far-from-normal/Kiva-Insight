import pandas as pd
from data_params import Data
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
import seaborn as sns

data_par = Data()
l1_ratio = data_par.l1_ratio
dir_to_saved_data = data_par.dir_to_saved_data

df_scores_list = []
df_reg_coef_list = []

for l1_rat in l1_ratio:
    path_to_scores = Path(dir_to_saved_data, "enet_" + str(l1_rat) + "_saved_score.csv")
    path_to_reg_coef = Path(dir_to_saved_data, str(l1_rat) + "_coefs_regularized.csv")

    df_scores = pd.read_csv(path_to_scores)
    df_scores["L1 ratio"] = l1_rat
    df_reg_coef = pd.read_csv(path_to_reg_coef)

    df_scores_list.append(df_scores)
    df_reg_coef_list.append(df_reg_coef)

    # df_regularize = pd.DataFrame({"L1 ratio": l1_ratio_str[:-1], "Number of coefficients": [num_coef], "Number of zero coefficients": [num_coef_0], "Ratio of zero": [100.0*num_coef_0_ratio]})

df_scores = pd.concat(df_scores_list)
df_scores.rename(columns={'Unnamed: 0': 'train-test'}, inplace=True)
df_scores.reset_index(inplace=True)
df_scores.drop(columns=["index"], inplace=True)
df_reg_coef = pd.concat(df_reg_coef_list)
df_reg_coef.rename(columns={'Unnamed: 0':' train-test'}, inplace=True)
df_reg_coef.reset_index(inplace=True)
df_reg_coef.drop(columns=["index"], inplace=True)
df_reg_coef["Percentage of non-zero coefficients"] = 100.0 - df_reg_coef["Ratio of zero"]
df_reg_coef["group"] = "1"
max_features = df_reg_coef["Number of coefficients"][0]
df_reg_coef
df_scores_precision = df_scores[["L1 ratio", "train-test", "precision"]]
df_scores_precision = df_scores_precision[~df_scores_precision["train-test"].isin(["train"])]
df_scores_precision
# %%
sns.set_style("ticks")


fig_dpi = 400
fig_size = (5, 3)
plt.figure(figsize=fig_size)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
ax = sns.lineplot(x="L1 ratio", y="precision",
    hue="train-test", style="train-test", alpha=0.75,
    markers=True, dashes=False, data=df_scores_precision)


ax.figure.tight_layout()
plt.legend(
    title="",
    # bbox_to_anchor=legend_coods_tags,
    frameon=False,)
handles, labels = ax.get_legend_handles_labels()
ax.legend(handles=handles[1:], labels=labels[1:])
ax.set(xlabel="L2 / ridge = 0                                           L1 / lasso = 1")
ax.set(ylabel="Precision")
sns.despine(offset = 10)
# ax.set(ylim=(0.97, 0.985))
# ax.set(ylim=(0.9, 1))

fig = ax.get_figure()
fig.savefig("precision_vs_regularization.png", dpi=fig_dpi)
plt.close()


# %%
fig_dpi = 400
fig_size = (5, 3)
plt.figure(figsize=fig_size)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
ax = sns.lineplot(x="L1 ratio", y="Percentage of non-zero coefficients",
    style="group", markers=True, data=df_reg_coef, legend=False)


ax.figure.tight_layout()
ax.set(xlabel="L2 / ridge = 0                                           L1 / lasso = 1")
ax.set(ylabel="Percentage of non-zero coefficients")
ax.set(ylim=(0, 100))
# ax.set(xlim=(0, 1))
ax.text(0.025, 95, "113,000 features", horizontalalignment='left', size='medium', color='black', weight='semibold')
ax.text(0.75, 25, "25,000 features", horizontalalignment='left', size='medium', color='black', weight='semibold')
sns.despine(offset = 10)
fig = ax.get_figure()
fig.savefig("percentage_of_coefs_equal_to_0_vs_regularization.png", dpi=fig_dpi)
plt.close()
