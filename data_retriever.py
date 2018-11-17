import configparser
import pandas as pd

def retrieveData():
    config = configparser.ConfigParser()
    config.read('init.cfg')
    path = config['RESOURCES']['path']
    dataFile = config['RESOURCES']['dataFile']

    df = pd.read_json(path + dataFile, lines=True)

    totalReviews = " "
    for review in df['reviewText'].items():
        totalReviews = totalReviews + review[1] + "\n"
    freviews = open(path + 'reviews.txt', 'w')
    freviews.write(totalReviews[1:-1])
    freviews.close()

    return (df, totalReviews)