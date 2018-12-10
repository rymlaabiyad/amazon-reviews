import pandas as pd


### Arguments : 
#   - path : The path of the file containing our data
### Output : 
#   - df : the dataframe corresponding to our data file
#   - reviews : An array containing all the reviews
def retrieveData(path):
    df = pd.read_json(path, lines=True)

    reviews = []
    for review in df['reviewText'].items():
        reviews.append(review[1])

    return (df, reviews)