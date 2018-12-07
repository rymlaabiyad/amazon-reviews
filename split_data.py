import pandas as pd
import configparser
from sklearn.model_selection import train_test_split

def get_test_train(df):
    config = configparser.ConfigParser()
    config.read('init.cfg')
    path = config['RESOURCES']['path']
    #df = pd.read_csv(path+'reviews.csv')
    train, test = train_test_split(df, test_size=0.2)
    train.to_csv(path+'train.csv')
    test.to_csv(path+'test.csv')
    return(train,test)


