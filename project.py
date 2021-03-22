# Importing packages that will be used throughout
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
import bokeh.palettes
import bokeh.plotting
import bokeh.io
from bokeh.plotting import figure, output_file
from bokeh.io import show
from bokeh.transform import factor_cmap
from bokeh.models import LinearInterpolator

# Importing CSV file into Pandas DataFrame
forbes = pd.read_csv("Forbes Top2000 2017.csv")
print(forbes.head())


# Dropping duplicates and missing values in DataFrame
# Firstly find if there a missing values from each column.
print(forbes.isna().sum())

# Finding the % of how much of the column is NA.
def my_func (a):
    percentage = (a / forbes.shape[0])*100
    return percentage

# To find, in a percentage, how much of the columns were NA using my custom function.
# First is the Sector column
print(my_func(197))

# Next is the Industry column
print(my_func(491))


# Dropping missing rows.
drop_rows = forbes.dropna()
print(drop_rows.shape)

# Using my custom function from earlier we will use it again here to find out much
# of the rows are left after excluding those rows with NA from above.
print(my_func(1508))

# This might have to do with a lot of the largest companies being involved in several sectors and industries.
# Therefore I don't think it is an error in the data.

# If needed to drop columns that include columns with missing data.
drop_columns = forbes.dropna(axis=1)
print(drop_columns.shape)

# To view if there is duplicates.
drop_duplicates = forbes.drop_duplicates(subset=['Company'])
print(drop_duplicates.shape[0])

# The duplicate company was 'Merck' which is based in US and Germany.

# Adding Company Name length as a column using for looping & itterows

for i, row in forbes.iloc[:21].iterrows():
        print(f"Index: {i}")
        print(f"{row['Company']}")



# Merging DataFrames. To find which country has the most companies in the top2000.
# Then finding out which country has the most per capita.

# First need to find the amount of unique countries. To find which populations to match the merging dataframe.

unique_countries = forbes["Country"].unique()
print(unique_countries)
print(forbes.nunique())

world_pop = pd.read_csv('countries of the world.csv')
world_pop['Country'] = world_pop['Country'].apply(lambda x: str(x).strip())

# Drop duplicates
print(world_pop.isna().sum())
drop_duplicates_of_world = world_pop.drop_duplicates(subset=['Country'])
print(drop_duplicates.shape[0])

# No Country and Population duplicates so can move onto the merging of dataframes.
merged_worldpop_and_forbes = forbes.merge(world_pop, left_on='Country', right_on='Country', how="outer")
print(merged_worldpop_and_forbes.head())
merged_worldpop_and_forbes.to_csv('merged_worldpop_and_forbes.csv')
pd.read_csv('merged_worldpop_and_forbes.csv', index_col=0)


# To find which country has the most per capita, we need to find how many and where the top 2000 companies are based.
frequency = forbes["Country"].value_counts()
print(frequency[0:61])

dict = {"Country":['China', 'United States', 'Japan', 'South Korea', 'Netherlands', 'Germany',
                   'Hong Kong', 'France', 'Spain', 'Switzerland', 'Brazil', 'Russia', 'Canada',
 'U.K', 'Australia', 'Taiwan', 'Italy', 'India', 'Ireland',
 'Saudi Arabia', 'Belgium', 'Sweden', 'Thailand', 'Luxembourg', 'Qatar',
 'Denmark', 'Singapore', 'Norway', 'United Arab Emirates', 'Mexico',
 'Indonesia', 'Malaysia', 'South Africa', 'Austria', 'Israel', 'Portugal',
 'Finland', 'Turkey', 'Colombia', 'Chile', 'Poland', 'Bermuda', 'Kuwait', 'Peru',
 'Philippines', 'Czech Rep.', 'Venezuela', 'Argentina', 'Hungary',
 'Morocco', 'Jordan', 'Bahrain', 'Lebanon', 'Mongolia', 'Nigeria', 'Oman',
 'Greece', 'Vietnam', 'Egypt', 'Pakistan', 'Puerto Rico'],
        'amount': [200, 564, 229, 64, 24, 51, 62, 59, 23, 46, 20, 27, 58, 91, 39, 46, 27, 58, 20, 17, 9, 26, 14,
          5, 8, 12, 17, 9, 14, 15, 6, 14, 10, 8, 1, 5, 9, 10, 6, 7, 5, 9, 3, 2, 8, 1, 3, 3, 2, 1,
          1, 2, 2, 1, 2, 1, 7, 4, 1, 1, 1],
        "Population":[1313973713, 298444215, 127463611, 48846823, 16491461, 82422299, 6940432, 60876136, 40397842,
                           7523934, 188078227, 142893540, 33098932, 60609153, 20264082, 23036087, 58133509, 1095351995,
                           4062235, 27019731, 10379067, 9016596, 64631595, 474413, 885359, 5450661, 4492150,
                           4610820, 2602713, 107449525, 245452739, 24385858, 44187637, 8192880, 6352117, 10605870,
                           5231372, 70413958, 43593035, 16134219, 38536869, 65773, 2418393, 28302603, 89468677,
                           10235455, 25730435, 39921833, 9981334, 33241259, 5906760, 698585, 3874050, 2832224,
                           131859731, 3102229, 10688058, 84402966, 78887007, 165803560, 3927188]}


data_dict = pd.DataFrame(dict)

# To find which country has the best ratio of population per top 2000 company in the country,
# we will be be creating a new column by a simple calculation and then finding the lowest amount in that column.
data_dict['capita_per_top2000'] = (data_dict['Population'] / data_dict['amount'])
print(data_dict[data_dict.capita_per_top2000 == data_dict.capita_per_top2000.min()])

# Bermuda has the most of the forbes top 2000 per person. With having one company per 7308 people there.


# Numpy show
# To show which company produces the best market value to sales
forbes = pd.read_csv("Forbes Top2000 2017.csv")
print(forbes["Market_Value"].isna().sum())
print(forbes["Sales"].isna().sum())
np_sales = np.array(forbes["Sales"])
np_mv = np.array(forbes["Market_Value"])
forbes['sales_to_marketval'] = (np_mv / np_sales)
print(forbes['Company'][forbes.sales_to_marketval == forbes.sales_to_marketval.max()])
print(forbes['sales_to_marketval'][forbes.sales_to_marketval == forbes.sales_to_marketval.max()])
# Results show that Porsche Automobile Holding produces the best sales to market value. With having 16100
# times their sales. Problem here is there is a feeling that this is an outlier.


# Seaborn and Matlplotlib
# Assess how European countries performed
fig, ax = plt.subplots(figsize=(13,7))
Europe = data_dict.iloc[[4,5,7,8,9,11,13,16,18,20,21,23,25,27,33,34,35,36,37,40,45,48,56]]
print(Europe)

ax.bar(Europe['Country'], Europe["amount"], color='b', edgecolor='black')
ax.set_xticklabels(Europe['Country'], rotation=70, fontsize=7)
ax.set_xlabel("Country")
ax.set_ylabel('frequency')
ax.set_title("Number of Europe's top 2000 Companies")
ax.tick_params(axis='x')
ax.tick_params(axis='y', colors='red')
plt.grid(color='#95a5a6', linestyle='--', linewidth=0.8, axis='y', alpha=0.4)
plt.savefig('Figure_1.png', dpi=500)
plt.show()
# U.K and Germany among some of the best performing, however the surprise would be Swizerland in this plot.

fig, ax = plt.subplots(2,1, sharex=True, figsize=(13, 7))
ax[0].plot(Europe["Country"], Europe["capita_per_top2000"])
ax[1].plot(Europe["Country"], Europe["Population"])
ax[0].set_ylabel("Company per capita", color='blue')
ax[1].set_ylabel("Population", color='red')
ax[1].set_xlabel("Country")
ax[1].set_xticklabels(Europe['Country'], rotation=70, fontsize=7)
ax[1].grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.5)
ax[0].grid(color='#95a5a6', linestyle='--', linewidth=1, axis='y', alpha=0.5)
plt.savefig('Figure_2.png', dpi=500)
plt.show()
# To assess Europe's capita per one of the top 2000 company two plots made on the same x-axis. The higher the
# population on the bottom graph but the lower the points on the upper means they would be performing the best.

fig, ax = plt.subplots(figsize=(13, 7))
Top100 = forbes[:101]
sns.scatterplot(data=Top100, x="Sales", y="Profits", hue="Sector", size='Market_Value')
plt.title('Top 100 of the Forbes top 2000 Sales and Profits')
sns.set(color_codes=True)
plt.savefig('Figure_3.png', dpi=500)
plt.show()



# Create the bokeh plot
size_mapper = LinearInterpolator(
    x=[Top100.Profits.min(), Top100.Profits.max()],
    y=[0, 50])
colors = bokeh.palettes.brewer['Spectral'][11],
plot = figure(plot_width=1100, plot_height=700, title = "Top 100 Companies & Sectors", toolbar_location=None,
          tools="hover", tooltips="@Company: (@Sales, @Market_Value, @Profits)")
index_cmap = factor_cmap('Sector', palette=colors[0],
                         factors=sorted(Top100.Sector.unique()))
plot.scatter('Sales','Market_Value',source=Top100, fill_alpha=0.9, fill_color=index_cmap,
             size={'field':'Profits','transform': size_mapper},legend='Sector')
plot.xaxis.axis_label = 'Sales'
plot.yaxis.axis_label = 'Market Value'
plot.legend.location = "top_right"
plot.legend.title = "Sector"
output_file("Figure_4.html")

show(plot)


# GeoPandas plots in Spyder
