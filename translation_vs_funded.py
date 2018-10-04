import pandas as pd
import numpy as np
from pathlib import Path
from data_params import Data

import matplotlib.pyplot as plt
import matplotlib
matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
import seaborn as sns

sns.set(style="darkgrid")

from data_utils import (
    preprocess_train_df,
    # fit_stats,
    # transform_stats,
    save_transformed_stats)

def summarize_stats(csv_name, dir_to_data, stat_name_select):

    path_to_input_df = Path(dir_to_data, csv_name)
    path_to_output_df = Path(dir_to_data, "funded_" + csv_name)

    df = pd.read_csv(path_to_input_df)

    df_funded = df.loc[df["STATUS"] == "funded"].drop(columns=["STATUS"])
    df_expired = df.loc[df["STATUS"] == "expired"].drop(columns=["STATUS"])

    df_funded_described = df_funded.describe(include="all")
    df_funded_described
    df_expired_described = df_expired.describe(include="all")
    df_expired_described

    funded_stat = df_funded_described.loc[stat_name_select]
    funded_stat = funded_stat.to_frame()
    funded_stat.reset_index(level=0, inplace=True)
    funded_stat.columns = ["FeatureName", "FeatureValueFunded"]

    expired_stat = df_expired_described.loc[stat_name_select]
    expired_stat = expired_stat.to_frame()
    expired_stat.reset_index(level=0, inplace=True)
    expired_stat.columns = ["FeatureName", "FeatureValueExpired"]

    funded_stat = funded_stat.merge(expired_stat, on="FeatureName")

    funded_stat.to_csv(path_to_output_df, index=False)

    return funded_stat


# %% ###########
data_par = Data()
cols_process = data_par.cols_process
cols_output = data_par.cols_output
valid_status = data_par.valid_status
dir_to_saved_data = data_par.dir_to_saved_data
dir_to_query_data = data_par.dir_to_query_data
path_to_training_data = data_par.path_to_training_data
stat_name_select = data_par.stat_name_select

predict = False

# csv_name_tags = "stats_tags_df.csv"
# csv_name_loanuse = "stats_loanuse_df.csv"
# csv_name_desc = "stats_desc_df.csv"

# df = pd.read_csv(
#     path_to_training_data, usecols=cols_process
#     , skiprows=lambda i: i % 10 != 0)
df = pd.read_csv(
    path_to_training_data, usecols=cols_process)

df = preprocess_train_df(df, valid_status, cols_output, predict)

df = df[["STATUS", "WAS_TRANSLATED", "ORIGINAL_LANGUAGE"]]
df = df[df["ORIGINAL_LANGUAGE"] != "None"]
df = df[df["WAS_TRANSLATED"] != np.nan]

df_status = df[["STATUS"]]
df_status = df_status.groupby(["STATUS"])["STATUS"].count()
df_status = pd.DataFrame(df_status)
df_status.columns = ["count"]
df_status = df_status.reset_index()
df_status["percent"] = 100 * df_status["count"] / df_status["count"].sum()
df_status

df_lang = df[["STATUS", "ORIGINAL_LANGUAGE"]].copy()
df_lang = df_lang.groupby(["STATUS", "ORIGINAL_LANGUAGE"])["STATUS"].count()
df_lang = df_lang.groupby(level=[1]).apply(lambda x: x / x.sum())
df_lang = df_lang.reset_index(level=[1])
df_lang.columns = ["ORIGINAL_LANGUAGE", "ratio"]
df_lang = df_lang.reset_index()
df_lang.columns = ["STATUS", "ORIGINAL_LANGUAGE", "ratio"]
df_lang = df_lang[df_lang["STATUS"] == "funded"]
df_lang = df_lang[["ORIGINAL_LANGUAGE", "ratio"]]
df_lang.columns = ["ORIGINAL_LANGUAGE", "percent_not_funded"]
df_lang["percent_not_funded"] = 100 * (1 - df_lang["percent_not_funded"])
df_lang = df_lang.reset_index().drop(columns=["index"])
df_lang = df_lang[df_lang["percent_not_funded"] != 0.0]
df_lang = df_lang.sort_values("percent_not_funded", ascending=False)
df_lang


df_trans = df[["STATUS", "WAS_TRANSLATED"]].copy()
df_trans = df_trans.groupby(["STATUS", "WAS_TRANSLATED"])["STATUS"].count()
df_trans = df_trans.groupby(level=[1]).apply(lambda x: x / x.sum())
df_trans = df_trans.reset_index(level=[1])
df_trans.columns = ["WAS_TRANSLATED", "ratio"]
df_trans = df_trans.reset_index()
df_trans.columns = ["STATUS", "WAS_TRANSLATED", "ratio"]
df_trans = df_trans[df_trans["STATUS"] == "funded"]
df_trans = df_trans[["WAS_TRANSLATED", "ratio"]]
df_trans.columns = ["WAS_TRANSLATED", "percent_expired"]
df_trans.columns = ["WAS_TRANSLATED", "percent_not_funded"]
df_trans["percent_not_funded"] = 100 * (1 - df_trans["percent_not_funded"])
# df_trans["WAS_TRANSLATED"] = int(df_trans["WAS_TRANSLATED"])
df_trans = df_trans.replace({0: "no", 1: "yes"})
df_trans = df_trans.reset_index().drop(columns=["index"])
df_trans


sns.set_style(
    "white",
    {
        "axes.spines.bottom": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": False,},)
fig_dpi = 400
fig_size = (4.5, 2.5)

plt.figure(figsize=fig_size)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
cnt_plot1 = sns.barplot(y="ORIGINAL_LANGUAGE", x="percent_not_funded", data=df_lang, palette=("Greens_r"))
for p in cnt_plot1.patches:
             cnt_plot1.annotate("%.1f" % p.get_width(), (p.get_width(), p.get_y() + p.get_height() / 2.),
                 ha='center', va='center', fontsize=11, color='white', xytext=(-11, 0),
                 textcoords='offset points', weight='semibold')

# for index, row in df_lang.iterrows():
#     print(row.ORIGINAL_LANGUAGE, row.percent_not_funded)
#     cnt_plot1.text(row.ORIGINAL_LANGUAGE, row.percent_not_funded, str(round(row.percent_not_funded,1)), color='black', ha="center")
# cnt_plot1.set_xticklabels(cnt_plot1.get_xticklabels(), rotation=0)
cnt_plot1.set(xticklabels=[])
cnt_plot1.set(xlabel="")
cnt_plot1.set(ylabel="")
cnt_plot1.set(title="Campaigns Not Funded (%)")
cnt_plot1.figure.tight_layout()
fig = cnt_plot1.get_figure()
fig.savefig("language_not_funded.png", dpi=fig_dpi)
plt.close()

plt.figure(figsize=(1.75, 1.75))
cnt_plot2 = sns.barplot(x="WAS_TRANSLATED", y="percent_not_funded", data=df_trans)
# cnt_plot2.set_xticklabels(cnt_plot2.get_xticklabels(), rotation=30)
cnt_plot2.figure.tight_layout()
cnt_plot2.set(xlabel="Translated?")
cnt_plot2.set(ylabel="Campaigns Not Funded (%)")
fig = cnt_plot2.get_figure()
fig.savefig("translated_not_funded.png", dpi=fig_dpi)
plt.close()
