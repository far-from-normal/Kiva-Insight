import pandas as pd
import numpy as np
from pathlib import Path
from data_params import Data
import matplotlib.pyplot as plt
import matplotlib
matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
import seaborn as sns
import pycountry



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

stats_desc = pd.read_csv(Path(dir_to_saved_data, "coefs_stats_df_desc.csv"))
stats_loanuse = pd.read_csv(Path(dir_to_saved_data, "coefs_stats_df_loanuse.csv"))
stats_tags = pd.read_csv(Path(dir_to_saved_data, "coefs_stats_df_tags.csv"))

words_desc = pd.read_csv(Path(dir_to_saved_data, "coefs_tfidf_df_desc.csv")).head(10)
words_loanuse = pd.read_csv(Path(dir_to_saved_data, "coefs_tfidf_df_loanuse.csv")).head(10)
words_tags = pd.read_csv(Path(dir_to_saved_data, "coefs_tfidf_df_tags.csv")).head(10)

loan_amount = pd.read_csv(Path(dir_to_saved_data, "coefs_df_loan_amount.csv")).head(10)
funding_days = pd.read_csv(Path(dir_to_saved_data, "coefs_df_funding_days.csv")).head(10)
was_translated = pd.read_csv(Path(dir_to_saved_data, "coefs_df_was_translated.csv")).head(10)

sector_code = pd.read_csv(Path(dir_to_saved_data, "coefs_df_sector_code.csv")).head(10)
sector_code['sector_code_names'] = sector_code['sector_code_names'].str[3:]

country_code = pd.read_csv(Path(dir_to_saved_data, "coefs_df_country_code.csv")).head(10)
country_code['country_code_names'] = country_code['country_code_names'].str[3:]
country_name_list = []
country_code_list = country_code["country_code_names"].values.tolist()
for code in country_code_list:
    country_name_list.append( pycountry.countries.get(alpha_2=code).name )
country_code["country_names"] = country_name_list

original_language = pd.read_csv(Path(dir_to_saved_data, "coefs_df_original_language.csv")).head(10)
original_language['original_language_names'] = original_language['original_language_names'].str[3:]

numerical_vars = pd.DataFrame(
    {
        "feature_names": [
            "loan_amount",
            "funding_days",
            "was_translated",],
        "coefs": [
            loan_amount["coefs"][0],
            funding_days["coefs"][0],
            was_translated["coefs"][0],]})
numerical_vars.sort_values("coefs", inplace=True, ascending=False)
numerical_vars

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


###########################

plt.figure(figsize=fig_size)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
cnt_plot1 = sns.barplot(y="stats_names", x="coefs", data=stats_desc, palette=("Greens_r"))
cnt_plot1.set(xlabel="")
cnt_plot1.set(ylabel="")
cnt_plot1.figure.tight_layout()
fig = cnt_plot1.get_figure()
fig.savefig("top_features_stats_desc.png", dpi=fig_dpi)
plt.close()



###########################

plt.figure(figsize=fig_size)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
cnt_plot1 = sns.barplot(y="stats_names", x="coefs", data=stats_loanuse, palette=("Greens_r"))
cnt_plot1.set(xlabel="")
cnt_plot1.set(ylabel="")
cnt_plot1.figure.tight_layout()
fig = cnt_plot1.get_figure()
fig.savefig("top_features_stats_loanuse.png", dpi=fig_dpi)
plt.close()




###########################

plt.figure(figsize=fig_size)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
cnt_plot1 = sns.barplot(y="stats_names", x="coefs", data=stats_tags, palette=("Greens_r"))
cnt_plot1.set(xlabel="")
cnt_plot1.set(ylabel="")
cnt_plot1.figure.tight_layout()
fig = cnt_plot1.get_figure()
fig.savefig("top_features_stats_tags.png", dpi=fig_dpi)
plt.close()



###########################

plt.figure(figsize=fig_size)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
cnt_plot1 = sns.barplot(y="tfidf_names", x="coefs", data=words_desc, palette=("Greens_r"))
cnt_plot1.set(xlabel="")
cnt_plot1.set(ylabel="")
cnt_plot1.figure.tight_layout()
fig = cnt_plot1.get_figure()
fig.savefig("top_features_words_desc.png", dpi=fig_dpi)
plt.close()




###########################

plt.figure(figsize=fig_size)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
cnt_plot1 = sns.barplot(y="tfidf_names", x="coefs", data=words_loanuse, palette=("Greens_r"))
cnt_plot1.set(xlabel="")
cnt_plot1.set(ylabel="")
cnt_plot1.figure.tight_layout()
fig = cnt_plot1.get_figure()
fig.savefig("top_features_words_loanuse.png", dpi=fig_dpi)
plt.close()





###########################

plt.figure(figsize=fig_size)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
cnt_plot1 = sns.barplot(y="tfidf_names", x="coefs", data=words_tags, palette=("Greens_r"))
cnt_plot1.set(xlabel="")
cnt_plot1.set(ylabel="")
cnt_plot1.figure.tight_layout()
fig = cnt_plot1.get_figure()
fig.savefig("top_features_words_tags.png", dpi=fig_dpi)
plt.close()


###########################
###########################
###########################
###########################

plt.figure(figsize=fig_size)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
cnt_plot1 = sns.barplot(y="feature_names", x="coefs", data=numerical_vars, palette=("Greens_r"))
cnt_plot1.set(xlabel="")
cnt_plot1.set(ylabel="")
cnt_plot1.figure.tight_layout()
fig = cnt_plot1.get_figure()
fig.savefig("top_features_numerical_vars.png", dpi=fig_dpi)
plt.close()



###########################

plt.figure(figsize=fig_size)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
cnt_plot1 = sns.barplot(y="sector_code_names", x="coefs", data=sector_code, palette=("Greens_r"))
# cnt_plot1.set(xticklabels=[])
cnt_plot1.set(xlabel="")
cnt_plot1.set(ylabel="")

cnt_plot1.figure.tight_layout()
fig = cnt_plot1.get_figure()
fig.savefig("top_features_sector_code.png", dpi=fig_dpi)
plt.close()




###########################

plt.figure(figsize=fig_size)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
cnt_plot1 = sns.barplot(y="country_names", x="coefs", data=country_code, palette=("Greens_r"))
cnt_plot1.set(xlabel="")
cnt_plot1.set(ylabel="")
cnt_plot1.figure.tight_layout()
fig = cnt_plot1.get_figure()
fig.savefig("top_features_country_code.png", dpi=fig_dpi)
plt.close()




###########################

plt.figure(figsize=fig_size)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
cnt_plot1 = sns.barplot(y="original_language_names", x="coefs", data=original_language, palette=("Greens_r"))
cnt_plot1.set(xlabel="")
cnt_plot1.set(ylabel="")
cnt_plot1.figure.tight_layout()
fig = cnt_plot1.get_figure()
fig.savefig("top_features_original_language.png", dpi=fig_dpi)
plt.close()
