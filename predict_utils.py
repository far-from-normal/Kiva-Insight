from pathlib import Path
import os
import pandas as pd
import numpy as np
from data_params import Data
import matplotlib.pyplot as plt
import seaborn as sns

from data_utils import (
    preprocess_train_df,
    fit_stats,
    transform_stats,
    save_transformed_stats,
    get_top_features,
)

# %%

def make_stripplot(fig_size, fig_dpi, legend_coods, marker_size, filename, data, loan_id):
    current_palette = sns.color_palette("Set1", n_colors=3)
    current_palette = [current_palette[2], current_palette[0], current_palette[1]]
    sns.set(font_scale=0.6)
    sns.set_palette(current_palette)
    sns.set_style(
        "white",
        {
            "axes.spines.bottom": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.grid": True,
        },
    )
    plt.figure(figsize=fig_size)
    sns_factor = sns.stripplot(
        y="Info",
        x="Performance",
        hue="Campaigns",
        data=data,
        size=marker_size,
        marker="D",
        alpha=.75,
        dodge=False,
        jitter=True,
        orient="v",
    )
    sns_factor.set(xticklabels=[])
    sns_factor.set(xlabel="")
    sns_factor.set(ylabel="")
    sns_factor.figure.tight_layout()
    sns_factor.legend(
        loc="upper center",
        ncol=3,
        title=None,
        bbox_to_anchor=legend_coods,
        frameon=False,
    )
    fig = sns_factor.get_figure()
    img_name = str(loan_id) + filename
    output_image_path_flask = Path("flaskexample", "static", img_name)
    fig.savefig(output_image_path_flask, dpi=fig_dpi)
    plt.close()
    return img_name

def text_suggestion(df, variable):
    unsuccessful = df.loc[df["Info"] == variable, "Unsuccessful"].values
    successful = df.loc[df["Info"] == variable, "Successful"].values
    yours = df.loc[df["Info"] == variable, "Your Campaign"].values
    print(unsuccessful)
    if successful > unsuccessful:
        if yours > successful:
            suggestion = "Decrease"
        elif yours < successful:
            suggestion = "Increase"
        else:
            suggestion = "Do not change"
    elif successful < unsuccessful:
        if yours > successful:
            suggestion = "Decrease"
        elif yours < successful:
            suggestion = "Increase"
        else:
            suggestion = "Do not change"
    else:
        suggestion = "Do not change"
    return suggestion


def normalize(x, x_min, x_max):
    return (x - x_min) / (x_max - x_min)


def normalize_df(df):
    # df_out = df.copy()
    for idx, row in df.iterrows():
        successful = row["Successful"]
        unsuccessful = row["Unsuccessful"]
        your_camp = row["Your Campaign"]
        if successful > unsuccessful:
            df.loc[idx, "Successful"] = 1.0
            df.loc[idx, "Unsuccessful"] = 0.0
            df.loc[idx, "Your Campaign"] = normalize(
                your_camp, unsuccessful, successful
            )
        elif successful < unsuccessful:
            df.loc[idx, "Successful"] = 0.0
            df.loc[idx, "Unsuccessful"] = 1.0
            df.loc[idx, "Your Campaign"] = normalize(
                your_camp, successful, unsuccessful
            )
        else:
            df.loc[idx, "Successful"] = 0.5
            df.loc[idx, "Unsuccessful"] = 0.5
            df.loc[idx, "Your Campaign"] = 0.5
    return df


def scrape_loan_id(loan_id):

    import data_params

    data_par = data_params.Data()
    unprocessed_vars = data_par.unprocessed_vars
    app_id = data_par.APP_ID

    scraped_file_name = Path("queries", "predict_scraped.csv")
    write_mode = "w+"

    # delete old clean file
    try:
        os.remove(scraped_file_name)
    except OSError:
        pass

    loan_id_str = str(loan_id)
    # print(loan_id_str)
    data_scraped, status, request_sucess = data_params.scrape_loan_data(
        loan_id_str, app_id
    )

    print(data_scraped)

    if request_sucess and status == "fundraising":
        data_formatted, df_formatted = data_params.format_scraped_data(
            data_scraped, status, unprocessed_vars
        )
        print(df_formatted)

        if not os.path.isfile(scraped_file_name):
            df_formatted.to_csv(
                scraped_file_name,
                header=unprocessed_vars,
                index=False,
                encoding="utf-8",
            )
        else:  # else it exists so append without writing the header
            df_formatted.to_csv(
                scraped_file_name,
                mode=write_mode,
                header=False,
                index=False,
                encoding="utf-8",
            )

    return request_sucess, status


def predict_prob():

    from sklearn.externals import joblib

    data_par = Data()
    cols_process = data_par.cols_process
    cols_output = data_par.cols_output
    valid_status = data_par.valid_status
    dir_to_saved_data = data_par.dir_to_saved_data
    dir_to_query_data = data_par.dir_to_query_data
    pipeline_type = data_par.prediction_model

    saved_data_dir = Path(dir_to_saved_data)
    trained_model_filename = saved_data_dir.joinpath(pipeline_type + "_model.pkl")
    pipeline = joblib.load(trained_model_filename)

    # %%
    input_data_filename = Path(dir_to_query_data, "predict_scraped.csv")
    df = pd.read_csv(input_data_filename, usecols=cols_process)
    df = preprocess_train_df(df, valid_status, cols_output, predict=True)
    prob = pipeline.predict_proba(df)

    return prob


def plot_factors(loan_id):

    data_par = Data()
    cols_process = data_par.cols_process
    cols_output = data_par.cols_output
    valid_status = data_par.valid_status
    dir_to_saved_data = data_par.dir_to_saved_data
    dir_to_query_data = data_par.dir_to_query_data

    # %%

    predict = True
    N_top_features = 6

    scraped_filename = Path(
        dir_to_query_data, "predict_scraped.csv"
    )  # input - predicted from scraping

    df = pd.read_csv(scraped_filename, usecols=cols_process)
    df = preprocess_train_df(df, valid_status, cols_output, predict)

    fit_stats(dir_to_saved_data, df)
    # returns all 3 transformers
    stats_tags_df, stats_loanuse_df, stats_desc_df = transform_stats(
        dir_to_saved_data, df
    )

    # tags
    # input
    feature_stats_filename_tags = Path(
        dir_to_saved_data, "funded_stats_tags_df.csv"
    )  # input - mean stats
    coefficient_ranking_filename_tags = Path(
        dir_to_saved_data, "coefs_stats_df_tags.csv"
    )  # input - coefficients
    # output
    predicted_features_filename_tags = str(loan_id) + "_predict_ranked_tags.csv"
    top_feature_compared_filename_tags = Path(
        dir_to_query_data, str(loan_id) + "_predict_compared_tags.csv"
    )  # input - predicted from scraping

    # loanuse
    # input
    feature_stats_filename_loanuse = Path(
        dir_to_saved_data, "funded_stats_loanuse_df.csv"
    )  # input - mean stats
    coefficient_ranking_filename_loanuse = Path(
        dir_to_saved_data, "coefs_stats_df_loanuse.csv"
    )  # input - coefficients
    # output
    predicted_features_filename_loanuse = str(loan_id) + "_predict_ranked_loanuse.csv"
    top_feature_compared_filename_loanuse = Path(
        dir_to_query_data, str(loan_id) + "_predict_compared_loanuse.csv"
    )  # input - predicted from scraping

    # desc
    # input
    feature_stats_filename_desc = Path(
        dir_to_saved_data, "funded_stats_desc_df.csv"
    )  # input - mean stats
    coefficient_ranking_filename_desc = Path(
        dir_to_saved_data, "coefs_stats_df_desc.csv"
    )  # input - coefficients
    # output
    predicted_features_filename_desc = str(loan_id) + "_predict_ranked_desc.csv"
    top_feature_compared_filename_desc = Path(
        dir_to_query_data, str(loan_id) + "_predict_compared_desc.csv"
    )  # input - predicted from scraping

    # transforming predicted statistical features
    save_transformed_stats(
        dir_to_query_data, stats_tags_df, predicted_features_filename_tags
    )
    save_transformed_stats(
        dir_to_query_data, stats_loanuse_df, predicted_features_filename_loanuse
    )
    save_transformed_stats(
        dir_to_query_data, stats_desc_df, predicted_features_filename_desc
    )

    # %%
    top_features_tags, top_features_long_tags = get_top_features(
        N_top_features,
        dir_to_query_data,
        predicted_features_filename_tags,
        feature_stats_filename_tags,
        coefficient_ranking_filename_tags,
        top_feature_compared_filename_tags,
    )
    top_features_loanuse, top_features_long_loanuse = get_top_features(
        N_top_features,
        dir_to_query_data,
        predicted_features_filename_loanuse,
        feature_stats_filename_loanuse,
        coefficient_ranking_filename_loanuse,
        top_feature_compared_filename_loanuse,
    )
    top_features_desc, top_features_long_desc = get_top_features(
        N_top_features,
        dir_to_query_data,
        predicted_features_filename_desc,
        feature_stats_filename_desc,
        coefficient_ranking_filename_desc,
        top_feature_compared_filename_desc,
    )

    # normalize data
    print(top_features_tags)
    print(top_features_loanuse)
    print(top_features_desc)

    top_features_tags = normalize_df(top_features_tags)
    top_features_loanuse = normalize_df(top_features_loanuse)
    top_features_desc = normalize_df(top_features_desc)

    print(top_features_tags.columns)
    # select top features
    top_features_tags = top_features_tags.loc[
        top_features_tags["Info"].isin(["num_words", "num_hashtags"])
    ]
    top_features_loanuse = top_features_loanuse.loc[
        top_features_loanuse["Info"].isin(["num_words"])
    ]
    top_features_desc = top_features_desc.loc[
        top_features_desc["Info"].isin(["num_words", "num_sentences", "num_paragraphs"])
    ]

    # make suggestion
    desc_text_words = text_suggestion(top_features_desc, "num_words")
    desc_text_sentences = text_suggestion(top_features_desc, "num_sentences")
    desc_text_paragraphs = text_suggestion(top_features_desc, "num_paragraphs")
    loanuse_text_words = text_suggestion(top_features_loanuse, "num_words")
    tags_text_words = text_suggestion(top_features_tags, "num_words")
    tags_text_hashtags = text_suggestion(top_features_tags, "num_words")

    top_features_tags = top_features_tags.replace(
        {"num_words": "number of words", "num_hashtags": "number of hashtags"}
    )
    top_features_loanuse = top_features_loanuse.replace(
        {"num_words": "number of words"}
    )
    top_features_desc = top_features_desc.replace(
        {
            "num_words": "number of words",
            "num_sentences": "number of sentences",
            "num_paragraphs": "number of paragraphs",
        }
    )

    print(top_features_tags)
    print(top_features_loanuse)
    print(top_features_desc)

    top_features_long_tags = pd.melt(
        top_features_tags,
        id_vars="Info",
        var_name="Campaigns",
        value_name="Performance",
    )
    top_features_long_loanuse = pd.melt(
        top_features_loanuse,
        id_vars="Info",
        var_name="Campaigns",
        value_name="Performance",
    )
    top_features_long_desc = pd.melt(
        top_features_desc,
        id_vars="Info",
        var_name="Campaigns",
        value_name="Performance",
    )

    fig_dpi = 400
    marker_size = 7

    legend_coods_tags = (0.3, 1.45)
    legend_coods_loanuse = (0.3, 1.65)
    legend_coods_desc = (0.3, 1.35)

    fig_size_tags = (3.5, 0.85)
    fig_size_loanuse = (3.5, 0.65)
    fig_size_desc = (3.5, 1)

    img_name_tags = make_stripplot(fig_size_tags, fig_dpi, legend_coods_tags, marker_size,
        "Campaign_tags.png", top_features_long_tags, loan_id)
    img_name_loanuse = make_stripplot(fig_size_loanuse, fig_dpi, legend_coods_loanuse, marker_size,
        "Campaign_loanuse.png", top_features_long_loanuse, loan_id)
    img_name_desc = make_stripplot(fig_size_desc, fig_dpi, legend_coods_desc, marker_size,
        "Campaign_desc.png", top_features_long_desc, loan_id)


    return (
        img_name_desc,
        img_name_loanuse,
        img_name_tags,
        desc_text_words,
        desc_text_sentences,
        desc_text_paragraphs,
        loanuse_text_words,
        tags_text_words,
        tags_text_hashtags,
    )
