import pandas as pd


# Arguments :
#   - path : The path of the file containing our data
# Output :
#   - df : the dataframe corresponding to our data file
#   - reviews : An array containing all the reviews
def retrieveData(maindf, asint='B005SUHPO6'):
    df = maindf[maindf.asin == asint]
    reviews = []
    for review in df['reviewText'].items():
        reviews.append(review[1])

    return (df, reviews)
