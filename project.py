# Importing packages that will be used throughout
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

# Importing CSV file into Pandas DataFrame
forbes = pd.read_csv("Forbes Top2000 2017.csv")
print(forbes.head())

# Dropping duplicates and missing values in DataFrame
# Firstly find if there a missing values from each column.
print(forbes.isna().sum())

# Finding the % of how much of the column is NA.
print((forbes["Industry"].isna().sum() / forbes.shape[0])*100)
print((forbes["Sector"].isna().sum() / forbes.shape[0])*100)

# Dropping missing rows.
drop_rows = forbes.dropna()
print(drop_rows.shape)

# Getting a % of how many would be left to use.
print((drop_rows.shape[0] / forbes.shape[0])*100)
# This might have to do with a lot of the largest companies being involved in several sectors and industries.
# Therefore I don't think it is an error in the data.

# If needed to drop columns that include columns with missing data.
drop_columns = forbes.dropna(axis=1)
print(drop_columns.shape)

# To view if there is duplicates.
drop_duplicates = forbes.drop_duplicates(subset=['Company'])
print(drop_duplicates.shape[0])

# The duplicate company was 'Merck' which is based in US and Germany.

country_based = forbes.sort_values("Country", ascending=True)


# Merging DataFrames. To find which country has the most companies in the top2000.
# Then finding out which country has the most per capita.

# First need to find the amount of unique countries. To find which populations to match the merging dataframe.

unique_countries = forbes["Country"].unique()
print(unique_countries)
print(forbes.nunique())

world_pop = pd.read_csv('countries of the world.csv')

# Drop duplicates
print(world_pop.isna().sum())
drop_duplicates_of_world = world_pop.drop_duplicates(subset=['Country'])
print(drop_duplicates.shape[0])

# No Country and Population duplicates so can move onto the merging of dataframes.
merged_worldpop_and_forbes = pd.merge(forbes,world_pop, left_on='Country', right_on='Country', how="outer")
print(merged_worldpop_and_forbes['Population'])
merged_worldpop_and_forbes.to_csv('merged_worldpop_and_forbes.csv')

frequency = merged_worldpop_and_forbes["Country"].value_counts()
#Unfinished merge ^

# Numpy show
# To show which company produces the best sales to profits turnover
print(forbes["Profits"].isna().sum())
print(forbes["Sales"].isna().sum())
np_sales = np.array(forbes["Sales"])
np_profits = np.array(forbes["Profits"])
sales_to_profits = (np_profits / np_sales) *100

merged_worldpop_and_forbes["sales_to_profits"] = (np.array(merged_worldpop_and_forbes["Sales"]) / np.array(merged_worldpop_and_forbes["Profits"])*100)
print(merged_worldpop_and_forbes)


merged_worldpop_and_forbes.groupby("Country")["sales_to_profits"].max()
