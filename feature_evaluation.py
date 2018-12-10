import pandas as pd
import numpy as np


### Arguments : 
#   - X : Each feature has a column containing whether it appears (1) or not (0) in a review
#   - y : The target, which has value 1 if the review had a good rating and 0 if not
### Output : A dataframe containing the occurence of each feature in positive and negative reviews
def evaluate(X, y):
    result = X
    result['rating'] = y
    positive_result = result[result['rating'] == 1]
    negative_result = result[result['rating'] == 0]
    positive_occurence = np.zeros(X.shape[1])
    negative_occurence = np.zeros(X.shape[1])
    for i in range(X.shape[1]):
        positive_occurence[i] = sum(positive_result.iloc[:,i])
        negative_occurence[i] = sum(negative_result.iloc[:,i])
    df_occ = pd.DataFrame(data={'feature' : X.columns.values, 'pos' : positive_occurence, 
                            'neg' : negative_occurence})

    return df_occ