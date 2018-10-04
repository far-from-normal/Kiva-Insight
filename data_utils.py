import pandas as pd
import numpy as np
import string
import re

# from datetime import datetime
from pathlib import Path

# import itertools
# import spacy
from html import unescape
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion

# from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text

# from sklearn.feature_extraction.text import CountVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib


def preprocess_train_df(df, valid_status, cols_output, predict=False):
    # moving empty DESCRIPTION_TRANSLATED from DESCRIPTION if DESCRIPTION_TRANSLATED is empty
    # df["DESCRIPTION_TRANSLATED"].fillna(df["DESCRIPTION"], inplace=True)
    # # converting NaN entries to empty strings
    # df["DESCRIPTION_TRANSLATED"].fillna("", inplace=True)
    # df["LOAN_USE"].fillna("", inplace=True)
    # df["TAGS"].fillna("", inplace=True)
    # df["SECTOR_NAME"].fillna("", inplace=True)
    # df["COUNTRY_CODE"].fillna("", inplace=True)

    df.loc[:, "DESCRIPTION_TRANSLATED"].fillna(df.loc[:, "DESCRIPTION"], inplace=True)
    # converting NaN entries to empty strings
    df.loc[:, "DESCRIPTION_TRANSLATED"].fillna("", inplace=True)
    df.loc[:, "LOAN_USE"].fillna("", inplace=True)
    df.loc[:, "TAGS"].fillna("", inplace=True)
    df.loc[:, "SECTOR_NAME"].fillna("", inplace=True)
    df.loc[:, "COUNTRY_CODE"].fillna("", inplace=True)
    df.loc[:, "ORIGINAL_LANGUAGE"].fillna("None", inplace=True)

    df["POSTED_TIME"] = pd.to_datetime(
        df["POSTED_TIME"], format="%Y-%m-%d %H:%M:%S.000 +0000"
    )
    df["PLANNED_EXPIRATION_TIME"] = pd.to_datetime(
        df["PLANNED_EXPIRATION_TIME"], format="%Y-%m-%d %H:%M:%S.000 +0000"
    )
    # subtracting Time
    df["FUNDING_DAYS"] = (
        df["PLANNED_EXPIRATION_TIME"] - df["POSTED_TIME"]
    ) / np.timedelta64(1, "D")

    # language / translation
    # df["WAS_TRANSLATED"] = np.nan
    df["WAS_TRANSLATED"] = 1
    df.loc[df["ORIGINAL_LANGUAGE"] == "English", "WAS_TRANSLATED"] = 0
    df.loc[df["ORIGINAL_LANGUAGE"] == "", "WAS_TRANSLATED"] = np.nan

    # removing the DESCRIPTION column
    # only keeping STAUTUS = expired and funded
    if predict is False:
        df = df[df["STATUS"].isin(valid_status)]
    # removing the unused coumns for training and prediction
    df = df[cols_output]
    # Final resetting of index
    df.reset_index(inplace=True, drop=True)

    return df


def save_transformed_stats(dir_to_saved_data, stats_df, csv_df_name):
    stats_df.to_csv(Path(dir_to_saved_data, csv_df_name), index=False)
    return None


def save_coefs(pipeline, dir_to_saved_data, classifier_type, l1_ratio_str):

    # %% get feature names
    feature_named_steps = pipeline.named_steps["features"].named_steps["feats"]
    stats_names_tags = (
        feature_named_steps.transformer_list[0][1]
        .named_steps["dictvect"]
        .get_feature_names()
    )
    stats_names_loanuse = (
        feature_named_steps.transformer_list[1][1]
        .named_steps["dictvect"]
        .get_feature_names()
    )
    stats_names_desc = (
        feature_named_steps.transformer_list[2][1]
        .named_steps["dictvect"]
        .get_feature_names()
    )
    tfidf_names_tags = (
        feature_named_steps.transformer_list[3][1]
        .named_steps["tfidf"]
        .get_feature_names()
    )
    tfidf_names_loanuse = (
        feature_named_steps.transformer_list[4][1]
        .named_steps["tfidf"]
        .get_feature_names()
    )
    tfidf_names_desc = (
        feature_named_steps.transformer_list[5][1]
        .named_steps["tfidf"]
        .get_feature_names()
    )

    n_stats_tags = len(stats_names_tags)
    n_stats_loanuse = len(stats_names_loanuse)
    n_stats_desc = len(stats_names_desc)
    n_tfidf_tags = len(tfidf_names_tags)
    n_tfidf_loanuse = len(tfidf_names_loanuse)
    n_tfidf_desc = len(tfidf_names_desc)

    n_feat_cumsum = [
        0,
        n_stats_tags,
        n_stats_loanuse,
        n_stats_desc,
        n_tfidf_tags,
        n_tfidf_loanuse,
        n_tfidf_desc,
    ]
    n_feat_cumsum = np.array(n_feat_cumsum).cumsum()
    idx_first = n_feat_cumsum[:-1].tolist()
    idx_last = n_feat_cumsum[1:].tolist()

    # %% get coefficients
    if classifier_type == "rf":
        coefs = pipeline.named_steps["clf"].feature_importances_
    else:  # use for lr or sgd
        # coefs = np.abs(pipeline.named_steps["clf"].coef_[0])
        coefs = pipeline.named_steps["clf"].coef_[0]
        num_coef_0 = np.sum(coefs == 0.0)
        num_coef = np.size(coefs)
        num_coef_0_ratio = num_coef_0 / num_coef
        print("######### Num coefs == 0:  ", num_coef_0, num_coef, num_coef_0_ratio)
        df_regularize = pd.DataFrame({"L1 ratio": l1_ratio_str[:-1], "Number of coefficients": [num_coef], "Number of zero coefficients": [num_coef_0], "Ratio of zero": [100.0*num_coef_0_ratio]})
        df_regularize.to_csv(
            Path(dir_to_saved_data, l1_ratio_str + "coefs_regularized.csv"), index=False
        )

    coefs_stats_tags = coefs[idx_first[0] : idx_last[0]]
    coefs_stats_loanuse = coefs[idx_first[1] : idx_last[1]]
    coefs_stats_desc = coefs[idx_first[2] : idx_last[2]]
    coefs_tfidf_tags = coefs[idx_first[3] : idx_last[3]]
    coefs_tfidf_loanuse = coefs[idx_first[4] : idx_last[4]]
    coefs_tfidf_desc = coefs[idx_first[5] : idx_last[5]]
    # make dataframes
    coefs_stats_df_tags = pd.DataFrame(
        {"stats_names": stats_names_tags, "coefs": coefs_stats_tags}
    )
    coefs_stats_df_loanuse = pd.DataFrame(
        {"stats_names": stats_names_loanuse, "coefs": coefs_stats_loanuse}
    )
    coefs_stats_df_desc = pd.DataFrame(
        {"stats_names": stats_names_desc, "coefs": coefs_stats_desc}
    )
    coefs_tfidf_df_tags = pd.DataFrame(
        {"tfidf_names": tfidf_names_tags, "coefs": coefs_tfidf_tags}
    )
    coefs_tfidf_df_loanuse = pd.DataFrame(
        {"tfidf_names": tfidf_names_loanuse, "coefs": coefs_tfidf_loanuse}
    )
    coefs_tfidf_df_desc = pd.DataFrame(
        {"tfidf_names": tfidf_names_desc, "coefs": coefs_tfidf_desc}
    )
    # sort dataframes
    coefs_stats_df_tags.sort_values("coefs", inplace=True, ascending=False)
    coefs_stats_df_loanuse.sort_values("coefs", inplace=True, ascending=False)
    coefs_stats_df_desc.sort_values("coefs", inplace=True, ascending=False)
    coefs_tfidf_df_tags.sort_values("coefs", inplace=True, ascending=False)
    coefs_tfidf_df_loanuse.sort_values("coefs", inplace=True, ascending=False)
    coefs_tfidf_df_desc.sort_values("coefs", inplace=True, ascending=False)
    # write dataframes
    coefs_stats_df_tags.to_csv(
        Path(dir_to_saved_data, l1_ratio_str + "coefs_stats_df_tags.csv"), index=False
    )
    coefs_stats_df_loanuse.to_csv(
        Path(dir_to_saved_data, l1_ratio_str + "coefs_stats_df_loanuse.csv"), index=False
    )
    coefs_stats_df_desc.to_csv(
        Path(dir_to_saved_data, l1_ratio_str + "coefs_stats_df_desc.csv"), index=False
    )
    coefs_tfidf_df_tags.to_csv(
        Path(dir_to_saved_data, l1_ratio_str + "coefs_tfidf_df_tags.csv"), index=False
    )
    coefs_tfidf_df_loanuse.to_csv(
        Path(dir_to_saved_data, l1_ratio_str + "coefs_tfidf_df_loanuse.csv"), index=False
    )
    coefs_tfidf_df_desc.to_csv(
        Path(dir_to_saved_data, l1_ratio_str + "coefs_tfidf_df_desc.csv"), index=False
    )

    print("Dimensionality of features:", coefs.shape)
    print("Dimensionality of features:", coefs_stats_tags.shape)
    print("Dimensionality of features:", coefs_stats_loanuse.shape)
    print("Dimensionality of features:", coefs_stats_desc.shape)

    print(coefs_stats_df_tags)
    print(coefs_tfidf_df_tags[:10])
    print(coefs_tfidf_df_tags[-10:])
    #
    print(coefs_stats_df_loanuse)
    print(coefs_tfidf_df_loanuse[:10])
    print(coefs_tfidf_df_loanuse[-10:])
    #
    print(coefs_stats_df_desc)
    print(coefs_tfidf_df_desc[:10])
    print(coefs_tfidf_df_desc[-10:])

    return None


def fit_stats(dir_to_fitted_transformer, df):

    path_to_fitted_transformer_tag = Path(dir_to_fitted_transformer, "stats_tag.pkl")
    path_to_fitted_transformer_loanuse = Path(
        dir_to_fitted_transformer, "stats_loanuse.pkl"
    )
    path_to_fitted_transformer_desc = Path(dir_to_fitted_transformer, "stats_desc.pkl")

    pipe_stats_tag, pipe_stats_loanuse, pipe_stats_desc = create_stats_pipeline()

    pipe_stats_tag.fit(df)
    pipe_stats_loanuse.fit(df)
    pipe_stats_desc.fit(df)

    joblib.dump(pipe_stats_tag, path_to_fitted_transformer_tag)
    joblib.dump(pipe_stats_loanuse, path_to_fitted_transformer_loanuse)
    joblib.dump(pipe_stats_desc, path_to_fitted_transformer_desc)

    return None


def transform_stats(dir_to_fitted_transformer, df):

    path_to_fitted_transformer_tag = Path(dir_to_fitted_transformer, "stats_tag.pkl")
    path_to_fitted_transformer_loanuse = Path(
        dir_to_fitted_transformer, "stats_loanuse.pkl"
    )
    path_to_fitted_transformer_desc = Path(dir_to_fitted_transformer, "stats_desc.pkl")

    pipe_stats_tag = joblib.load(path_to_fitted_transformer_tag)
    pipe_stats_loanuse = joblib.load(path_to_fitted_transformer_loanuse)
    pipe_stats_desc = joblib.load(path_to_fitted_transformer_desc)

    # print(pipe_stats_tag.named_steps["stats"].get_feature_names())

    stats_tags = pipe_stats_tag.transform(df)
    stats_tags_df = pd.DataFrame(
        stats_tags, columns=pipe_stats_tag.named_steps["stats"].get_feature_names()
    )
    stats_tags_df["STATUS"] = df["STATUS"]

    stats_loanuse = pipe_stats_loanuse.transform(df)
    stats_loanuse_df = pd.DataFrame(
        stats_loanuse,
        columns=pipe_stats_loanuse.named_steps["stats"].get_feature_names(),
    )
    stats_loanuse_df["STATUS"] = df["STATUS"]

    stats_desc = pipe_stats_desc.transform(df)
    stats_desc_df = pd.DataFrame(
        stats_desc, columns=pipe_stats_desc.named_steps["stats"].get_feature_names()
    )
    stats_desc_df["STATUS"] = df["STATUS"]

    return stats_tags_df, stats_loanuse_df, stats_desc_df


def get_top_features(
    N_top_features,
    dir_to_query_data,
    predicted_features_filename,
    feature_stats_filename,
    coefficient_ranking_filename,
    top_feature_compared_filename,
):

    # %%
    feature_stats = pd.read_csv(feature_stats_filename)
    coefficient_ranking = pd.read_csv(coefficient_ranking_filename)

    predicted_features = pd.read_csv(
        Path(dir_to_query_data, predicted_features_filename)
    )
    predicted_features.drop(["STATUS"], axis=1, inplace=True)
    predicted_features = predicted_features.T
    predicted_features.reset_index(level=0, inplace=True)
    predicted_features.columns = ["FeatureName", "FeatureValuePredicted"]

    coefficient_ranking.columns = ["FeatureName", "coefs"]
    top_N_features = coefficient_ranking.loc[0:N_top_features, "FeatureName"]

    top_stats = feature_stats.loc[feature_stats["FeatureName"].isin(top_N_features)]
    top_predicted = predicted_features.loc[
        predicted_features["FeatureName"].isin(top_N_features)
    ]

    top_features = top_stats.merge(top_predicted, on="FeatureName")
    top_features.reset_index(level=0, inplace=True, drop=True)
    top_features.columns = ["Info", "Successful", "Unsuccessful", "Your Campaign"]

    top_features.to_csv(top_feature_compared_filename, index=False, encoding="utf-8")

    top_features_long = pd.melt(
        top_features, id_vars="Info", var_name="Campaigns", value_name="Performance"
    )

    return top_features, top_features_long


def preprocess_tfidf(doc):
    # nlp = spacy.load('en')
    # fixed bug: TypeError: argument of type 'float' is not iterable (in unescape)
    if not isinstance(doc, str):
        doc = ""
    doc = unescape(doc)
    doc = doc.replace("\\n", " ")
    doc = doc.replace("\\r", " ")
    doc = doc.replace("\n", " ")
    translator = re.compile("[%s]" % re.escape(string.punctuation))
    doc = translator.sub(" ", doc)
    word_list = [word.lower() for word in doc.split()]
    # word_list = [word.lower() if word in nlp.vocab else "" for word in doc.split()]
    word_list = " ".join(word_list)
    return word_list


def preprocess_count(doc):
    # fixed bug: TypeError: argument of type 'float' is not iterable (in unescape)
    if not isinstance(doc, str):
        doc = ""
    doc = unescape(doc)
    doc = doc.replace("\\n", " ")
    doc = doc.replace("\\r", " ")
    doc = doc.replace("\n", " ")
    translator = re.compile("[%s]" % re.escape(string.punctuation))
    doc = translator.sub(" ", doc)
    word_list = [word.lower() for word in doc.split()]
    return word_list


class DataFrameColumnExtracter(BaseEstimator, TransformerMixin):
    """
    see http://scikit-learn.org/stable/auto_examples/hetero_feature_union.html
    """

    def __init__(self, column):
        self.column = column

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.column]


class OneColumnReshaper(BaseEstimator, TransformerMixin):
    """
    If features are only one column, reshape to 2D array.
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X.values.reshape(-1, 1)


class TextStats(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self

    def transform(self, docs):
        stats_list = []
        translator = re.compile("[%s]" % re.escape(string.punctuation))

        for text in docs:
            # print(text)
            # fixed bug: TypeError: argument of type 'float' is not iterable (in unescape)
            if not isinstance(text, str):
                text = ""
            no_html = unescape(text)
            no_html_no_punct = translator.sub(" ", no_html)
            words = no_html_no_punct.split(" ")

            # bolded_text_type1 =

            # num_bold_text = text.count("<strong>") + text.count("<b>")
            num_dollar_signs = no_html.count("$")
            num_hashtags = no_html.count("#")
            num_paragraphs = (
                1
                + (
                    text.count("\\r\\n")
                    + text.count("\r\n")
                    + text.count("<br /><br />")
                    + text.count("<p>")
                ),
            )  # text.count("\n") + text.count("\n\n")
            num_sentences = text.count(".")
            num_words = len(words)

            stats = {
                # "num_bold_parts": num_bold_text,
                "num_dollar_signs": num_dollar_signs,
                "num_hashtags": num_hashtags,
                "num_paragraphs": num_paragraphs,
                "num_sentences": num_sentences,
                "num_words": num_words,
            }
            stats_list.append(stats)

        return stats_list

    def get_feature_names(self):
        return [
            # "num_bold_parts",
            "num_dollar_signs",
            "num_hashtags",
            "num_paragraphs",
            "num_sentences",
            "num_words",
        ]


def create_stats_pipeline():
    # %% Text Stats Pipelines
    pipe_stats_tag = Pipeline(
        [
            ("extractor", DataFrameColumnExtracter(column="TAGS")),
            ("stats", TextStats()),
            ("dictvect", DictVectorizer(sparse=False)),
        ]
    )
    pipe_stats_loanuse = Pipeline(
        [
            ("extractor", DataFrameColumnExtracter(column="LOAN_USE")),
            ("stats", TextStats()),
            ("dictvect", DictVectorizer(sparse=False)),
        ]
    )
    pipe_stats_desc = Pipeline(
        [
            ("extractor", DataFrameColumnExtracter(column="DESCRIPTION_TRANSLATED")),
            ("stats", TextStats()),
            ("dictvect", DictVectorizer(sparse=False)),
        ]
    )

    return pipe_stats_tag, pipe_stats_loanuse, pipe_stats_desc


def create_pipeline(classifier_type, l1_ratio):
    extra_words = [
        "translated",
        "Translated",
        "original",
        "language",
        "volunteer",
        "description"]
    my_stop_words = text.ENGLISH_STOP_WORDS.union(extra_words)
    # %% Text Stats Pipelines
    pipe_stats_tag = Pipeline(
        [
            ("extractor", DataFrameColumnExtracter(column="TAGS")),
            ("stats", TextStats()),
            ("dictvect", DictVectorizer(sparse=False)),
            ("scaler", StandardScaler()),
        ]
    )
    pipe_stats_loanuse = Pipeline(
        [
            ("extractor", DataFrameColumnExtracter(column="LOAN_USE")),
            ("stats", TextStats()),
            ("dictvect", DictVectorizer(sparse=False)),
            ("scaler", StandardScaler()),
        ]
    )
    pipe_stats_desc = Pipeline(
        [
            ("extractor", DataFrameColumnExtracter(column="DESCRIPTION_TRANSLATED")),
            ("stats", TextStats()),
            ("dictvect", DictVectorizer(sparse=False)),
            ("scaler", StandardScaler()),
        ]
    )
    # %% Tfidf Pipelines
    pipe_tfidf_tag = Pipeline(
        [
            ("extractor", DataFrameColumnExtracter(column="TAGS")),
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=preprocess_tfidf,
                    ngram_range=(1, 1),
                    stop_words=my_stop_words,
                    # stop_words="english",
                    sublinear_tf=True,
                    strip_accents='unicode',
                    norm="l1",
                    max_features=100000,
                ),
            ),
        ]
    )
    pipe_tfidf_loanuse = Pipeline(
        [
            ("extractor", DataFrameColumnExtracter(column="LOAN_USE")),
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=preprocess_tfidf,
                    ngram_range=(1, 1),
                    stop_words=my_stop_words,
                    # stop_words="english",
                    sublinear_tf=True,
                    strip_accents='unicode',
                    norm="l1",
                    max_features=100000,
                ),
            ),
        ]
    )
    pipe_tfidf_desc = Pipeline(
        [
            ("extractor", DataFrameColumnExtracter(column="DESCRIPTION_TRANSLATED")),
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=preprocess_tfidf,
                    ngram_range=(1, 1),
                    stop_words=my_stop_words,
                    # stop_words="english",
                    sublinear_tf=True,
                    strip_accents='unicode',
                    norm="l1",
                    max_features=100000,
                ),
            ),
        ]
    )
    pipe_loan_amount = Pipeline(
        [
            ("extractor", DataFrameColumnExtracter(column="LOAN_AMOUNT")),
            ("reshape2D", OneColumnReshaper()),
            ("imputation", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    pipe_funding_days = Pipeline(
        [
            ("extractor", DataFrameColumnExtracter(column="FUNDING_DAYS")),
            ("reshape2D", OneColumnReshaper()),
            ("imputation", SimpleImputer(missing_values=np.nan, strategy="mean")),
            ("scaler", StandardScaler()),
        ]
    )
    pipe_was_translated = Pipeline(
        [
            ("extractor", DataFrameColumnExtracter(column="WAS_TRANSLATED")),
            ("reshape2D", OneColumnReshaper()),
            ("imputation", SimpleImputer(missing_values=np.nan, strategy="mean")),
            # ("scaler", StandardScaler()),
        ]
    )
    pipe_sector_name = Pipeline(
        [
            ("extractor", DataFrameColumnExtracter(column="SECTOR_NAME")),
            ("reshape2D", OneColumnReshaper()),
            # ("labelenc", LabelEncoder()),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
            # ("labelbin", LabelBinarizer()),
            ("imputation", SimpleImputer(missing_values=np.nan, strategy="constant")),
        ]
    )
    pipe_country_code = Pipeline(
        [
            ("extractor", DataFrameColumnExtracter(column="COUNTRY_CODE")),
            ("reshape2D", OneColumnReshaper()),
            # ("labelenc", LabelEncoder()),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
            # ("labelbin", LabelBinarizer()),
            ("imputation", SimpleImputer(missing_values=np.nan, strategy="constant")),
        ]
    )
    pipe_original_language = Pipeline(
        [
            ("extractor", DataFrameColumnExtracter(column="ORIGINAL_LANGUAGE")),
            ("reshape2D", OneColumnReshaper()),
            # ("labelenc", LabelEncoder()),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
            # ("labelbin", LabelBinarizer()),
            ("imputation", SimpleImputer(missing_values=np.nan, strategy="constant")),
        ]
    )

    # %% features feature union
    fu_feats = FeatureUnion(
        # fu_feats = ColumnTransformer(
        transformer_list=[
            ("stats_tag", pipe_stats_tag),
            ("stats_loanuse", pipe_stats_loanuse),
            ("stats_desc", pipe_stats_desc),
            ("tfidf_tag", pipe_tfidf_tag),
            ("tfidf_loanuse", pipe_tfidf_loanuse),
            ("tfidf_desc", pipe_tfidf_desc),
            ("loan_amount", pipe_loan_amount),
            ("funding_days", pipe_funding_days),
            ("was_translated", pipe_was_translated),
            ("sector_name", pipe_sector_name),
            ("country_code", pipe_country_code),
            ("original_language", pipe_original_language),
        ],
        transformer_weights={
            "stats_tag": 1.,
            "stats_loanuse": 1.,
            "stats_desc": 1.,
            "tfidf_tag": 1.,
            "tfidf_loanuse": 1.,
            "tfidf_desc": 1.,
            "loan_amount": 1.,
            "funding_days": 1.,
            "was_translated": 1.,
            "sector_code": 1.,
            "country_code": 1.,
            "original_language": 1.,
        },
    )

    # %% feature pipeline
    pipe_feats = Pipeline([("feats", fu_feats)])

    # %% classifier pipelines

    if classifier_type == "lr":
        pipeline = Pipeline(
            [
                ("features", pipe_feats),
                (
                    "clf",
                    LogisticRegression(
                        class_weight="balanced", solver="saga"  # "liblinear" "saga"
                    ),
                ),
            ]
        )
    elif classifier_type == "enet":  # "l2", "l1",
        pipeline = Pipeline(
            [
                ("features", pipe_feats),
                (
                    "clf",
                    SGDClassifier(
                        class_weight="balanced",
                        loss="log",
                        penalty="elasticnet",
                        l1_ratio=l1_ratio, #0.15
                    ),
                ),
            ]
        )
    else:  # rf
        pipeline = Pipeline(
            [
                ("features", pipe_feats),
                (
                    "clf",
                    RandomForestClassifier(
                        class_weight="balanced", n_estimators=250, random_state=42
                    ),
                ),
            ]
        )

    return pipeline
