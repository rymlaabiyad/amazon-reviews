import nltk
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

def extract(reviews):
    # Retrieve the 300 most common relevent words 
    token_count = tokenize(reviews, 300)
    tokens = []
    for token in token_count:
        tokens.append(token[0])
        
    # Retrieve the nouns out of the 300 most common words
    nouns = tag(tokens, 'NN')
    print('nouns : ', nouns[0:30])
    #return nouns[0:30]
    
    
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
    
