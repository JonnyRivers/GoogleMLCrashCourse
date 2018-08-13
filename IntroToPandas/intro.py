# https://colab.research.google.com/notebooks/mlcc/intro_to_pandas.ipynb
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")

import numpy as np
import pandas as pd

city_names = pd.Series(['San Francisco', 'San Jose', 'Scaramento'])
population = pd.Series([852469, 1015785, 485199])

cities_dataframe = pd.DataFrame({'City name': city_names, 'Population': population})

print(cities_dataframe.describe()) # summary stats
print(type(cities_dataframe['City name']))
print(cities_dataframe['City name'])

california_housing_dataframe = pd.read_csv("https://storage.googleapis.com/mledu-datasets/california_housing_train.csv", sep=",")
print(california_housing_dataframe.describe()) # summary stats

print(california_housing_dataframe.head()) # first few records

print(california_housing_dataframe.hist('housing_median_age')) # histogram

print(population / 1000)
print(np.log(population))

large_pop = population.apply(lambda val: val > 1000000)
print (large_pop)

cities_dataframe['Area square miles'] = pd.Series([46.87, 176.53, 97.92])
cities_dataframe['Population density'] = cities_dataframe['Population'] / cities_dataframe['Area square miles']
print(cities_dataframe)

cities_dataframe['Is wide and has saint name'] = cities_dataframe['City name'].apply(lambda val: val.startswith('San')) & (cities_dataframe['Area square miles'] > 50)
print(cities_dataframe)

print(city_names.index)
print(cities_dataframe.index)

reindexed_cities_dataframe = cities_dataframe.reindex([3,0,1,2,])
print(reindexed_cities_dataframe)