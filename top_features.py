import pandas as pd
import numpy as np
from pathlib import Path
from data_params import Data
import matplotlib.pyplot as plt
import matplotlib
matplotlib.font_manager.findSystemFonts(fontpaths=None, fontext='ttf')
import seaborn as sns



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
# for p in cnt_plot1.patches:
#              cnt_plot1.annotate("%.1f" % p.get_width(), (p.get_width(), p.get_y() + p.get_height() / 2.),
#                  ha='center', va='center', fontsize=11, color='white', xytext=(-11, 0),
#                  textcoords='offset points', weight='semibold')

# cnt_plot1.set(xticklabels=[])
cnt_plot1.set(xlabel="")
cnt_plot1.set(ylabel="")
# cnt_plot1.set(title="Campaigns Not Funded (%)")
cnt_plot1.figure.tight_layout()
fig = cnt_plot1.get_figure()
fig.savefig("top_features_stats_desc.png", dpi=fig_dpi)
plt.close()



###########################

plt.figure(figsize=fig_size)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
cnt_plot1 = sns.barplot(y="stats_names", x="coefs", data=stats_loanuse, palette=("Greens_r"))
# for p in cnt_plot1.patches:
#              cnt_plot1.annotate("%.1f" % p.get_width(), (p.get_width(), p.get_y() + p.get_height() / 2.),
#                  ha='center', va='center', fontsize=11, color='white', xytext=(-11, 0),
#                  textcoords='offset points', weight='semibold')

# cnt_plot1.set(xticklabels=[])
cnt_plot1.set(xlabel="")
cnt_plot1.set(ylabel="")
# cnt_plot1.set(title="Campaigns Not Funded (%)")
cnt_plot1.figure.tight_layout()
fig = cnt_plot1.get_figure()
fig.savefig("top_features_stats_loanuse.png", dpi=fig_dpi)
plt.close()




###########################

plt.figure(figsize=fig_size)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
cnt_plot1 = sns.barplot(y="stats_names", x="coefs", data=stats_tags, palette=("Greens_r"))
# for p in cnt_plot1.patches:
#              cnt_plot1.annotate("%.1f" % p.get_width(), (p.get_width(), p.get_y() + p.get_height() / 2.),
#                  ha='center', va='center', fontsize=11, color='white', xytext=(-11, 0),
#                  textcoords='offset points', weight='semibold')

# cnt_plot1.set(xticklabels=[])
cnt_plot1.set(xlabel="")
cnt_plot1.set(ylabel="")
# cnt_plot1.set(title="Campaigns Not Funded (%)")
cnt_plot1.figure.tight_layout()
fig = cnt_plot1.get_figure()
fig.savefig("top_features_stats_tags.png", dpi=fig_dpi)
plt.close()



###########################

plt.figure(figsize=fig_size)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
cnt_plot1 = sns.barplot(y="tfidf_names", x="coefs", data=words_desc, palette=("Greens_r"))
# for p in cnt_plot1.patches:
#              cnt_plot1.annotate("%.1f" % p.get_width(), (p.get_width(), p.get_y() + p.get_height() / 2.),
#                  ha='center', va='center', fontsize=11, color='white', xytext=(-11, 0),
#                  textcoords='offset points', weight='semibold')

# cnt_plot1.set(xticklabels=[])
cnt_plot1.set(xlabel="")
cnt_plot1.set(ylabel="")
# cnt_plot1.set(title="Campaigns Not Funded (%)")
cnt_plot1.figure.tight_layout()
fig = cnt_plot1.get_figure()
fig.savefig("top_features_words_desc.png", dpi=fig_dpi)
plt.close()




###########################

plt.figure(figsize=fig_size)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
cnt_plot1 = sns.barplot(y="tfidf_names", x="coefs", data=words_loanuse, palette=("Greens_r"))
# for p in cnt_plot1.patches:
#              cnt_plot1.annotate("%.1f" % p.get_width(), (p.get_width(), p.get_y() + p.get_height() / 2.),
#                  ha='center', va='center', fontsize=11, color='white', xytext=(-11, 0),
#                  textcoords='offset points', weight='semibold')

# cnt_plot1.set(xticklabels=[])
cnt_plot1.set(xlabel="")
cnt_plot1.set(ylabel="")
# cnt_plot1.set(title="Campaigns Not Funded (%)")
cnt_plot1.figure.tight_layout()
fig = cnt_plot1.get_figure()
fig.savefig("top_features_words_loanuse.png", dpi=fig_dpi)
plt.close()





###########################

plt.figure(figsize=fig_size)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = 'Arial'
cnt_plot1 = sns.barplot(y="tfidf_names", x="coefs", data=words_tags, palette=("Greens_r"))
# for p in cnt_plot1.patches:
#              cnt_plot1.annotate("%.1f" % p.get_width(), (p.get_width(), p.get_y() + p.get_height() / 2.),
#                  ha='center', va='center', fontsize=11, color='white', xytext=(-11, 0),
#                  textcoords='offset points', weight='semibold')

# cnt_plot1.set(xticklabels=[])
cnt_plot1.set(xlabel="")
cnt_plot1.set(ylabel="")
# cnt_plot1.set(title="Campaigns Not Funded (%)")
cnt_plot1.figure.tight_layout()
fig = cnt_plot1.get_figure()
fig.savefig("top_features_words_tags.png", dpi=fig_dpi)
plt.close()
