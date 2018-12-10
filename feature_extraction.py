import nltk
from nltk.util import ngrams

#########################################################
##                                                     ##
##              IMPORTANT                              ##
## If it's the first time you run the program, please  ##
## uncomment the following lines :                     ##
##                                                     ##
#########################################################

#nltk.dowload('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('averaged_perceptron_tagger')

from collections import Counter


### Arguments : 
#   - reviews : An array containing all the reviews
### Output : The most interesting tokens in the reviews 
def extract(reviews):
    totalReviews = " "
    for review in reviews:
        totalReviews = totalReviews + review + "\n"
        
    # Retrieve the 300 most common relevent words 
    token_count = tokenize(totalReviews, 300)
    tokens = [t[0] for t in token_count]
        
    # Retrieve the nouns out of the 300 most common words
    nouns = tag(tokens, 'NN')
    print('nouns : ', nouns[0:30])
    #adjectives=tag(tokens,'JJ')
    #return nouns[0:30], adjectives[0:20]
    return nouns[0:30]
    
    
### Arguments : 
#   - reviews : string to tokenize
#   - nb_tokens : number of tokens we want to return
### Output : This method returns a list of tuples, containing the nb_tokens (or 100, 
#   as a default value) most common words in a text, excluding stop words ("and", "the", etc)
def tokenize(reviews, nb_tokens=100):
    #We start with a string containg all the reviews
    #We retrieve all the words, then only keep alphabetic ones (no numbers)
    tokens = nltk.tokenize.word_tokenize(reviews)
    alpha_tokens = [t for t in tokens if t.isalpha()] 
    #We get rid of capital letters, in order to count word occurence properly
    lower_tokens = [t.lower() for t in alpha_tokens]

    #We get rid of stop words 
    stop_words = set(nltk.corpus.stopwords.words('english'))
    no_stops_tokens = [t for t in lower_tokens if t not in(stop_words)]

    wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_tokens = [wordnet_lemmatizer.lemmatize(t) for t in no_stops_tokens]
    count = Counter(lemmatized_tokens)
    
    return count.most_common(nb_tokens)
    

### Arguments : 
#   - tokens : words to classify
#   - tag : The tag used to classify. Specific values accepted : ('VB', 'NN', 'JJ', .. )
#   see nltk documentation for more details
### Output : This method returns a sublist of 'tokens' which were tagged with 'tag'
def tag(tokens, tag):
    tagged = nltk.pos_tag(tokens)
    return [t[0] for t in tagged if t[1] == tag]
    

### Arguments : 
#   - tokens : words to stem/lemmetize
### Output : This method returns a list containing the root word of each token
def stem(tokens):
    ps = nltk.PorterStemmer()
    return [ps.stem(t) for t in tokens]


def filter(review, features):
    tokens = [t[0] for t in tokenize(review)]
    filtered_tokens = [t for t in tokens if t in features]
    return filtered_tokens


###############################################################################
    
def get_filter_token(text,common):
    txt_tk=nltk.tokenize.word_tokenize(text, language='english')
    stop_words = nltk.corpus.stopwords.words("english")
    stop_words.append(':)')
    stop_words.append('....')
    alpha_tokens = [t for t in txt_tk if t.isalpha()]
    lower_tokens = [t.lower() for t in alpha_tokens]
    no_stops_tokens = [t for t in lower_tokens if t not in (stop_words)]
    wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_tokens = [wordnet_lemmatizer.lemmatize(t) for t in no_stops_tokens]
    filt_tokens_cmn=[f for f in lemmatized_tokens if f in common]
    filt_tokens_cmn=set(filt_tokens_cmn)
    return list(filt_tokens_cmn)

def get_token(text):
    txt_tk=nltk.tokenize.word_tokenize(text, language='english')
    stop_words = nltk.corpus.stopwords.words("english")
    stop_words.append(':)')
    stop_words.append('....')
    alpha_tokens = [t for t in txt_tk if t.isalpha()]
    lower_tokens = [t.lower() for t in alpha_tokens]
    no_stops_tokens = [t for t in lower_tokens if t not in (stop_words)]
    wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_tokens = [wordnet_lemmatizer.lemmatize(t) for t in no_stops_tokens]
    return list(lemmatized_tokens)

def get_tag(text):
    pos = {}
    pos = nltk.pos_tag(text)
    return pos

def get_cmn_nlp_features(df,cmn_nn,cmn_adj):
    df['tokens'] = df.apply(lambda row:get_token(row['reviewText']),axis =1)
    df['tokens_cmn_nn'] = df.apply(lambda row:get_filter_token(row['reviewText'],cmn_nn),axis =1)
    df['tokens_cmn_adj'] = df.apply(lambda row: get_filter_token(row['reviewText'], cmn_adj), axis=1)
    df['sents_length'] = df.apply(lambda row: len(row['tokens']), axis=1)
    df['tokens_pos'] = df.apply(lambda row: get_tag(row['tokens']), axis=1)
    df['aspects'] = df.apply(lambda row: get_aspects(row['tokens_pos']), axis=1)
    return df

#def get_aspects1(text):
#    inputTupples = dict(text)
#    return (len(inputTupples))
def get_aspects(text):
    inputTupples=dict(text)
    prevWord = ''
    prevTag = ''
    currWord = ''
    aspectList = []
    outputDict = {}
    # Extracting Aspects
    #for key,value in inputTupples.items():
    for word, tag in inputTupples.items():
            if (tag == 'NN' or tag == 'NNP'):
                if (prevTag == 'NN' or prevTag == 'NNP'):
                        currWord = prevWord + '-' + word
                else:
                        aspectList.append(prevWord.lower())
                        currWord = word
            prevWord = currWord
            prevTag = tag

    return (aspectList)

