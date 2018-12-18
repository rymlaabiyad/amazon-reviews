from data_retriever import retrieveData
from data_prep import prepareData
from classify import classify, featureanalysis
from feature_evaluation import evaluate
from DataViz import wordmap
from configparser import ConfigParser
from sklearn.model_selection import train_test_split
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


config = ConfigParser()
config.read('init.cfg')
path = config['RESOURCES']['path']
dataFile = config['RESOURCES']['dataFile']
maindf = pd.read_json(path + dataFile, lines=True)


# We retrieve a dataframe corresponding to our data file
# and an array containing only the reviews for a given product, here B000S5Q9CA
df, reviews = retrieveData(maindf, 'B000S5Q9CA')
print('df.shape : ', df.shape)


# Our target data y is related to product ratings
# Our input data is whether a feature is mentioned or not in a product's review
# If a review contains a feature, the corresponding column will have value 1, else 0.
y, X = prepareData(df['overall'], reviews)

# Printing a wordmap for positive features
wordmap(df, 1)
# First frequency analysis
df_occ = evaluate(X, y)
print(df_occ)


# We split the data intro training/testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=0, stratify=y)

# We run our classifier (the first parameter indicates which classifier to use)
# Choose among : 'lr', 'svc', 'dt' and 'nb'

bestclf, accuracy_score = classify(
    'lr', X_train, y_train, X_test, y_test)
print('accuracy score for the chosen classifier: ', accuracy_score)

# We train the classifier on the whole dataset :
bestclf.fit(X, y)

# Selection of the most relevant features for the classifier (does not work for naive bayes classifier)
featureanalysis(X, bestclf)
