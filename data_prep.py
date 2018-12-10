from feature_extraction import extract, filter
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing
import pandas as pd
import numpy as np


def prepareData(ratings, reviews):
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
    
    x_df = pd.DataFrame(x.toarray(), columns=cv.get_feature_names())
    
    # To simplify our classification, we make sure that high ratings are considered positive (1),
    # while low ones are considered negative (0)
    ratings = transform_rating(ratings)
            
    return ratings, x_df

def transform_rating(ratings):
    # First, we normalize the data
    ratings = preprocessing.scale(ratings)
    #Then we only keep positive (1) and negative (0) ratings, to simplify classification
    mean = ratings.mean()
    transformed_ratings = np.zeros(ratings.shape[0])
    for i in range(ratings.shape[0]):
        if(ratings[i] > mean):
            transformed_ratings[i] = 1
        else:
            transformed_ratings[i] = 0
    return transformed_ratings