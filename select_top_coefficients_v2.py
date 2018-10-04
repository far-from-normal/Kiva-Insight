import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import (
    preprocess_train_df,
    fit_stats,
    transform_stats,
    save_transformed_stats,
)

from data_params import Data

# %%
sns.set(style="darkgrid", font_scale=2)

data_par = Data()
cols_process = data_par.cols_process
valid_status = data_par.valid_status
dir_to_saved_data = data_par.dir_to_saved_data
dir_to_query_data = data_par.dir_to_query_data
path_to_training_data = data_par.path_to_training_data

predict = False

N_top_features = 6
loan_id = 44

feature_stats_filename = Path("saved_data", "stats_tags_df.csv")  # input - mean stats
coefficient_ranking_filename = Path(
    "saved_data", "coefs_stats_df_desc.csv"
)  # input - coefficients
scraped_filename = Path(
    "queries", "predict_scraped.csv"
)  # input - predicted from scraping
predicted_features_filename = str(loan_id) + "_predict_ranked_tags.csv"  # output

df = pd.read_csv(scraped_filename, usecols=cols_process)
df = preprocess_train_df(df, valid_status, predict)

fit_stats(scraped_filename, df)
# returns all 3 transformers
stats_tags_df, stats_loanuse_df, stats_desc_df = transform_stats(dir_to_saved_data, df)
save_transformed_stats("queries", stats_tags_df, predicted_features_filename)

# %%

feature_stats = pd.read_csv(feature_stats_filename)
coefficient_ranking = pd.read_csv(coefficient_ranking_filename)

# %%
predicted_features = pd.read_csv(predicted_features_filename)
predicted_features.drop(["STATUS"], axis=1, inplace=True)
predicted_features = predicted_features.T
predicted_features.reset_index(level=0, inplace=True)
predicted_features.columns = ["FeatureName", "FeatureValuePredicted"]

top_N_features = coefficient_ranking.loc[0:N_top_features, "FeatureName"]

top_stats = feature_stats.loc[feature_stats["FeatureName"].isin(top_N_features)]
top_predicted = predicted_features.loc[
    predicted_features["FeatureName"].isin(top_N_features)
]

# %%
top_features = top_stats.merge(top_predicted, on="FeatureName")
top_features.reset_index(level=0, inplace=True, drop=True)
top_features.columns = ["Info", "Successful", "Unsuccessful", "Your Campaign"]
top_features

# top_features["d"] = top_features["Successful"] - top_features["uccessful"]

top_features_long = pd.melt(
    top_features, id_vars="Info", var_name="Campaigns", value_name="Performance"
)

sns_factor = sns.catplot(
    y="Info",
    x="Performance",
    hue="Campaigns",
    data=top_features_long,
    kind="bar",
    height=10,
    aspect=1.2,
)

fig = plt.gcf()
fig = sns_factor.fig
fig.savefig("Campaign.png", dpi=300)
