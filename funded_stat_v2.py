import pandas as pd
import numpy as np
from pathlib import Path
from data_params import Data


from data_utils import (
    preprocess_train_df,
    fit_stats,
    transform_stats,
    save_transformed_stats,
)


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

csv_name_tags = "stats_tags_df.csv"
csv_name_loanuse = "stats_loanuse_df.csv"
csv_name_desc = "stats_desc_df.csv"

df = pd.read_csv(path_to_training_data, usecols=cols_process)

df = preprocess_train_df(df, valid_status, cols_output, predict)

fit_stats(dir_to_saved_data, df)
stats_tags_df, stats_loanuse_df, stats_desc_df = transform_stats(dir_to_saved_data, df)
save_transformed_stats(dir_to_saved_data, stats_tags_df, csv_name_tags)
save_transformed_stats(dir_to_saved_data, stats_loanuse_df, csv_name_loanuse)
save_transformed_stats(dir_to_saved_data, stats_desc_df, csv_name_desc)

funded_df_tags = summarize_stats(csv_name_tags, dir_to_saved_data, stat_name_select)
funded_df_loanuse = summarize_stats(
    csv_name_loanuse, dir_to_saved_data, stat_name_select
)
funded_df_desc = summarize_stats(csv_name_desc, dir_to_saved_data, stat_name_select)

funded_df_tags
funded_df_tags.to_csv(Path(dir_to_saved_data, "funded_df_tags.csv"), index=False)
funded_df_loanuse
funded_df_loanuse.to_csv(Path(dir_to_saved_data, "funded_df_loanuse.csv"), index=False)
funded_df_desc
funded_df_desc.to_csv(Path(dir_to_saved_data, "funded_df_desc.csv"), index=False)
