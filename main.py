from data_retriever import retrieveData
from feature_extraction import extract, filter

from configparser import ConfigParser
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd


config = ConfigParser()
config.read('init.cfg')
path = config['RESOURCES']['path']
dataFile = config['RESOURCES']['dataFile']

# We retrieve a dataframe corresponding to our data file
# and an array containing only the reviews
df, reviews = retrieveData(path + dataFile)

# We call the extract method to retrieve the most relevant words (features)
features = extract(reviews)  
# For each review, we only keep the words that are features
filtered_tokens = [filter(review, features) for review in reviews]
filtered_reviews = []
for f in filtered_tokens:
    review = ""
    for t in f:
        review = review + t + " "
    filtered_reviews.append(review)

# We create a column for each feature
# If that feature is mentioned in the review : the value is 1, else 0
cv = CountVectorizer(binary=True)
x = cv.fit_transform(filtered_reviews)
df_features = pd.DataFrame(x.toarray(), columns=features)
print(df_features.info())