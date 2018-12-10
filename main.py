from data_retriever import retrieveData
from data_prep import prepareData

from configparser import ConfigParser
import pandas as pd


config = ConfigParser()
config.read('init.cfg')
path = config['RESOURCES']['path']
dataFile = config['RESOURCES']['dataFile']

# We retrieve a dataframe corresponding to our data file
# and an array containing only the reviews
df, reviews = retrieveData(path + dataFile)
print(df.shape)

# We add to it columns. One for each feature (relevant word).
# If a review contains a feature, the corresponding column will have value 1, else 0.
df_features = pd.concat([df, prepareData(df, reviews)], axis=1)
print(df_features.shape)
#json = df_features.to_json(orient='records')
#f = open(path + 'features.json', 'w')
#f.write(json[1:-1])
#f.close()