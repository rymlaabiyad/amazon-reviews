from data_retriever import retrieveData
from feature_extraction import *
from split_data import *
df, reviews = retrieveData()

cmn_nn,cmn_adj = extract(reviews)

train,test=get_test_train(df=df)
print('Training DataSet Shape:',train.shape)
print('Test DataSet Shape:',test.shape)
print(cmn_nn)
print(cmn_adj)
train=get_cmn_nlp_features(df=train,cmn_nn=cmn_nn,cmn_adj=cmn_adj)
#print(train['tokenized_sents'].head())

config = configparser.ConfigParser()
config.read('init.cfg')
path = config['RESOURCES']['path']
#df = pd.read_csv(path+'reviews.csv')

train.to_csv(path+'train.csv')
test.to_csv(path+'test.csv')