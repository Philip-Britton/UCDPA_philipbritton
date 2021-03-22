#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 17:32:34 2021

@author: philipbritton
"""

import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

counties = gpd.read_file('counties/counties.shp')
print(counties.head())
counties.plot(figsize=(14,14), legend=True)


listings = pd.read_csv('updated_listings5.csv')
print(listings['longitude'])
sns.scatterplot(data=listings, x='longitude', y='latitude', hue='county')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.show()


# To focus in on an area of the graph. In this instance it will be Dublin
import geopandas as gpd
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

counties = gpd.read_file('counties/counties.shp')
print(counties.head())
counties.plot(figsize=(14,14), legend=True)


listings = pd.read_csv('updated_listings5.csv')
print(listings['longitude'])
sns.scatterplot(data=listings, x='longitude', y='latitude', hue='county', 
                size='size')
plt.legend(loc='center left', bbox_to_anchor=(1.0, 0.5))
plt.xlim(-6.6, -5.9)
plt.ylim(53.2, 53.7)
plt.show()

# Using centroid and chainging the epsg to flat surfaces
counties['NAME_TAG']
print(counties.centroid)
print(counties.crs)
counties.geometry = counties.geometry.to_crs(epsg=3857)
counties['centre'] = counties.centroid
print(counties['centre'])



import folium
counties = counties.to_crs(epsg=4326)
dublin = counties.iloc[27]
dublin_centre = dublin.centre
dub = folium.Map(location=[53.39222, -6.28398], zoom_start=9)

folium.GeoJson(dublin.geometry).add_to(dub)
dub.save("output.html")
display(dub)

