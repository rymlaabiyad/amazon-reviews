from data_retriever import retrieveData
from feature_extraction import extract

df, reviews = retrieveData()

features = extract(reviews)