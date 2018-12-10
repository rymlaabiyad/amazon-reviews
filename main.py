from data_retriever import retrieveData
from data_prep import prepareData
from classify import classify
from feature_evaluation import evaluate

from configparser import ConfigParser
from sklearn.model_selection import train_test_split


config = ConfigParser()
config.read('init.cfg')
path = config['RESOURCES']['path']
dataFile = config['RESOURCES']['dataFile']

# We retrieve a dataframe corresponding to our data file
# and an array containing only the reviews
df, reviews = retrieveData(path + dataFile)
print('df.shape : ', df.shape)

# Our target data y is related to product ratings
# Our input data is whether a feature is mentioned or not in a product's review
# If a review contains a feature, the corresponding column will have value 1, else 0.
y, X = prepareData(df['overall'], reviews)

# We split the data intro training/testing data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

# We run our classifier (the first parameter indicates which classifier to use)
y_pred, accuracy_score = classify('lr', X_train, y_train, X_test, y_test)
print('accuracy score : ', accuracy_score)

df_occ = evaluate(X_test, y_pred)
print(df_occ)