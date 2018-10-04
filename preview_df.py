import pandas as pd
import numpy as np
from collections import Counter

import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="darkgrid", font_scale=2)

n = 1  # every 100th line = 1% of the lines
df = pd.read_csv("loans.csv", skiprows=lambda i: i % n != 0)
list(df)
df = df[
    [
        "STATUS",
        "ACTIVITY_NAME",
        "SECTOR_NAME",
        "POSTED_TIME",
        "RAISED_TIME",
        "FUNDED_AMOUNT",
    ]
]
df = df.loc[df["STATUS"].isin(["expired", "funded"]), :]

df[["POSTED_TIME", "RAISED_TIME"]] = df[["POSTED_TIME", "RAISED_TIME"]].apply(
    pd.to_datetime, format="%Y-%m-%d %H:%M:%S.%f"
)

df["RAISED_INTERVAL"] = (df["RAISED_TIME"] - df["POSTED_TIME"]).dt.days
# df["RAISED_INTERVAL"] = df["RAISED_INTERVAL"].dt.day
df.drop(columns=["POSTED_TIME", "RAISED_TIME"], inplace=True)

raised_interval = df["RAISED_INTERVAL"]
raised_interval = raised_interval[~raised_interval.isnull()]
raised_interval = raised_interval[raised_interval > 0]
sns.distplot(raised_interval)

funded_amount = df["FUNDED_AMOUNT"]
funded_amount = funded_amount[~funded_amount.isnull()]
funded_amount = funded_amount[funded_amount > 0]
sns.distplot(funded_amount)

df_activity = pd.DataFrame(
    df.groupby("STATUS")["ACTIVITY_NAME"].value_counts()
)  # .reset_index(inplace=True)
df_activity.columns = ["counts"]
df_activity.reset_index(inplace=True)

df_activity_wide = df_activity.pivot(
    index="ACTIVITY_NAME", columns="STATUS", values="counts"
).fillna(0)
df_activity_wide.reset_index(inplace=True)
df_activity_wide["ratio"] = df_activity_wide["funded"] / (
    df_activity_wide["expired"] + df_activity_wide["funded"]
)
df_activity_wide.sort_values("ratio", axis=0, ascending=False, inplace=True)

sns_factor = sns.catplot(
    y="ACTIVITY_NAME",
    x="ratio",
    data=df_activity_wide,
    kind="bar",
    height=30,
    aspect=0.6,
)


fig = plt.gcf()
fig = sns_factor.fig
fig.savefig("Activity.png", dpi=300)

# df_funded_amount
# df_interval_raised

# df_long = pd.melt(df, id_vars="Info")#, var_name="STATUS", value_name="Performance")
# df_long = pd.melt(df, id_vars=["STATUS"], var_name="Variable", value_name="Value")
# df_long

df_sector = pd.DataFrame(
    df.groupby("STATUS")["SECTOR_NAME"].value_counts().fillna(0)
)  # .reset_index(inplace=True)
df_sector.columns = ["counts"]
df_sector.reset_index(inplace=True)

df_sector_wide = df_sector.pivot(
    index="SECTOR_NAME", columns="STATUS", values="counts"
).fillna(0)
# df_sector_wide.drop(columns=["STATUS"], inplace=True)
df_sector_wide.reset_index(inplace=True)
df_sector_wide["ratio"] = df_sector_wide["funded"] / (
    df_sector_wide["expired"] + df_sector_wide["funded"]
)
df_sector_wide["total"] = df_sector_wide["expired"] + df_sector_wide["funded"]
df_sector_wide.sort_values("ratio", axis=0, ascending=False, inplace=True)

sns_factor = sns.catplot(
    y="SECTOR_NAME", x="ratio", data=df_sector_wide, kind="bar", height=10, aspect=1.2
)

fig = plt.gcf()
fig = sns_factor.fig
fig.savefig("Sector.png", dpi=300)


sns_factor = sns.catplot(
    y="SECTOR_NAME", x="total", data=df_sector_wide, kind="bar", height=10, aspect=1.2
)

fig = plt.gcf()
fig = sns_factor.fig
fig.savefig("Sector_total.png", dpi=300)
