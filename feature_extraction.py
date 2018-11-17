import nltk
from collections import Counter

def extract(reviews):
    tokenize(reviews)
    
    #TODO : extract features
    
def tokenize(reviews):
    #We start with a string containg all the reviews
    #We retrieve all the words, then only keep alphabetic ones (no numbers)
    tokens = nltk.tokenize.word_tokenize(reviews)
    alpha_tokens = [t for t in tokens if t.isalpha()] 
    #We get rid of capital letters, in order to count word occurence properly
    lower_tokens = [t.lower() for t in alpha_tokens]

    #We get rid of stop words such as "and", "the" ...
    stop_words = set(nltk.corpus.stopwords.words('english'))
    no_stops_tokens = [t for t in lower_tokens if t not in(stop_words)]

    wordnet_lemmatizer = nltk.stem.WordNetLemmatizer()
    lemmatized_tokens = [wordnet_lemmatizer.lemmatize(t) for t in no_stops_tokens]
    count = Counter(lemmatized_tokens)
    
    print(count.most_common(20))