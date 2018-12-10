from feature_extraction import extract, filter
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

def prepareData(df, reviews):
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
    return pd.DataFrame(x.toarray(), columns=features)