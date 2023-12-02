# Project 3 Report: Movie Review Sentiment Analysis
# CS 598 Practical Statistical Learning, Fall 2023 
#
# Team Members 
# * Kurt Tuohy (ktuohy): contraction expansion, token t-test, token exploration
# * Neal Ryan (nealpr2): notebook configuration, initial tokenization, lasso token regularization
# * Alelign Faris (faris2): mymain.py
#
# Approach based on Campuswire post 628.
#
# TO RUN THIS SCRIPT:
# Call "python mymain.py" with three arguments:
# 1) myvocab_file: the name of the file with desired vocabulary, e.g. "myvocab.txt"
# 2) train_file: the training data file, e.g. "train.tsv"
# 3) test_file: the test data file, e.g. "test.tsv"


import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
#from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score


SEED = 4031
np.random.seed(SEED)

def expand_contractions(reviews):

    """
    Routine to expand English contractions, like "isn't" --> "is not".
    This is because "isn't good" and "wasn't good" will both expand to produce the bi-gram "not good".
    The pooled phrase should have more predictive power than the original two phrases.
    """

    # Dictionary of English contractions. Taken from StackOverflow post, which borrowed it from Wikipedia:
    # https://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python

    contractions = { 
        "\\bain't\\b": "am not",
        "\\baren't\\b": "are not",
        "\\bcan't\\b": "cannot",
        "\\bcan't've\\b": "cannot have",
        "\\b'cause\\b": "because",
        "\\bcould've\\b": "could have",
        "\\bcouldn't\\b": "could not",
        "\\bcouldn't've\\b": "could not have",
        "\\bdidn't\\b": "did not",
        "\\bdoesn't\\b": "does not",
        "\\bdon't\\b": "do not",
        "\\bhadn't\\b": "had not",
        "\\bhadn't've\\b": "had not have",
        "\\bhasn't\\b": "has not",
        "\\bhaven't\\b": "have not",
        "\\bhe'd\\b": "he would",
        "\\bhe'd've\\b": "he would have",
        "\\bhe'll\\b": "he will",
        "\\bhe'll've\\b": "he will have",
        "\\bhe's\\b": "he is",
        "\\bhow'd\\b": "how did",
        "\\bhow'd'y\\b": "how do you",
        "\\bhow'll\\b": "how will",
        "\\bhow's\\b": "how is",
        "\\bi'd\\b": "i would",
        "\\bi'd've\\b": "i would have",
        "\\bi'll\\b": "i will",
        "\\bi'll've\\b": "i will have",
        "\\bi'm\\b": "i am",
        "\\bi've\\b": "i have",
        "\\bisn't\\b": "is not",
        "\\bit'd\\b": "it would",
        "\\bit'd've\\b": "it would have",
        "\\bit'll\\b": "it will",
        "\\bit'll've\\b": "it will have",
        "\\bit's\\b": "it is",
        "\\blet's\\b": "let us",
        "\\bma'am\\b": "madam",
        "\\bmayn't\\b": "may not",
        "\\bmight've\\b": "might have",
        "\\bmightn't\\b": "might not",
        "\\bmightn't've\\b": "might not have",
        "\\bmust've\\b": "must have",
        "\\bmustn't\\b": "must not",
        "\\bmustn't've\\b": "must not have",
        "\\bneedn't\\b": "need not",
        "\\bneedn't've\\b": "need not have",
        "\\bo'clock\\b": "of the clock",
        "\\boughtn't\\b": "ought not",
        "\\boughtn't've\\b": "ought not have",
        "\\bshan't\\b": "shall not",
        "\\bsha'n't\\b": "shall not",
        "\\bshan't've\\b": "shall not have",
        "\\bshe'd\\b": "she would",
        "\\bshe'd've\\b": "she would have",
        "\\bshe'll\\b": "she will",
        "\\bshe'll've\\b": "she will have",
        "\\bshe's\\b": "she is",
        "\\bshould've\\b": "should have",
        "\\bshouldn't\\b": "should not",
        "\\bshouldn't've\\b": "should not have",
        "\\bso've\\b": "so have",
        "\\bso's\\b": "so is",
        "\\bthat'd\\b": "that would",
        "\\bthat'd've\\b": "that would have",
        "\\bthat's\\b": "that is",
        "\\bthere'd\\b": "there would",
        "\\bthere'd've\\b": "there would have",
        "\\bthere's\\b": "there is",
        "\\bthey'd\\b": "they would",
        "\\bthey'd've\\b": "they would have",
        "\\bthey'll\\b": "they will",
        "\\bthey'll've\\b": "they will have",
        "\\bthey're\\b": "they are",
        "\\bthey've\\b": "they have",
        "\\bto've\\b": "to have",
        "\\bwasn't\\b": "was not",
        "\\bwe'd\\b": "we would",
        "\\bwe'd've\\b": "we would have",
        "\\bwe'll\\b": "we will",
        "\\bwe'll've\\b": "we will have",
        "\\bwe're\\b": "we are",
        "\\bwe've\\b": "we have",
        "\\bweren't\\b": "were not",
        "\\bwhat'll\\b": "what will",
        "\\bwhat'll've\\b": "what will have",
        "\\bwhat're\\b": "what are",
        "\\bwhat's\\b": "what is",
        "\\bwhat've\\b": "what have",
        "\\bwhen's\\b": "when is",
        "\\bwhen've\\b": "when have",
        "\\bwhere'd\\b": "where did",
        "\\bwhere's\\b": "where is",
        "\\bwhere've\\b": "where have",
        "\\bwho'll\\b": "who will",
        "\\bwho'll've\\b": "who will have",
        "\\bwho's\\b": "who is",
        "\\bwho've\\b": "who have",
        "\\bwhy's\\b": "why is",
        "\\bwhy've\\b": "why have",
        "\\bwill've\\b": "will have",
        "\\bwon't\\b": "will not",
        "\\bwon't've\\b": "will not have",
        "\\bwould've\\b": "would have",
        "\\bwouldn't\\b": "would not",
        "\\bwouldn't've\\b": "would not have",
        "\\by'all\\b": "you all",
        "\\by'all'd\\b": "you all would",
        "\\by'all'd've\\b": "you all would have",
        "\\by'all're\\b": "you all are",
        "\\by'all've\\b": "you all have",
        "\\byou'd\\b": "you would",
        "\\byou'd've\\b": "you would have",
        "\\byou'll\\b": "you will",
        "\\byou'll've\\b": "you will have",
        "\\byou're\\b": "you are",
        "\\byou've\\b": "you have"
    }
    
    # Replace all contractions in all reviews.
    for contraction in contractions:
        reviews = reviews.str.replace(contraction, contractions[contraction], regex=True)
        #reviews = re.sub(contraction, contractions[contraction], reviews)

    return reviews

def preprocess_reviews(reviews):
    """
    Routine to preprocess text: strip out HTML, convert to lowercase, and expand English contractions.
    """
    # Remove HTML tags
    reviews = reviews.str.replace('<.*?>', ' ', regex=True)
    # Convert to lowercase
    reviews = reviews.str.lower()
    # Expand English contractions
    reviews = expand_contractions(reviews)
    
    return reviews

def main():
    #parser = argparse.ArgumentParser(description="Process three input files.")

    #parser.add_argument('myvocab_file', type=str, help='Path to myvocab.txt')
    #parser.add_argument('train_file', type=str, help='Path to train.tsv')
    #parser.add_argument('test_file', type=str, help='Path to test.tsv')

    #args = parser.parse_args()

    #vocab = args.myvocab_file
    #train = args.train_file
    #test = args.test_file

    vocab = 'myvocab.txt'
    train = 'train.tsv'
    test = 'test.tsv'

    vocab = pd.read_csv(vocab, sep=',', header=None, names=['word'])
    best_vocab = vocab['word'].tolist()
    train = pd.read_csv(train, sep='\t')
    test = pd.read_csv(test, sep='\t')

    stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "their", "they", "his", \
             "her", "she", "he", "a", "an", "and", "is", "was", "are", "were", "him", "himself", "has", "have", "it", "its", \
             "the", "us"]

    top_feature_vectorizer = CountVectorizer(
        vocabulary=best_vocab,          # The top 1000 features
        stop_words=stopwords,             # Remove stop words
        ngram_range=(1, 4),               # Use 1- to 4-grams
        min_df=0.001,                     # Minimum term frequency
        max_df=0.5,                       # Maximum document frequency
        token_pattern=r"\b[\w+\|']+\b"    # Use word tokenizer, but don't split on apostrophes
    )

    print("Tokenizing Reviews (~1 min)...")
    dtm_vocab_train = top_feature_vectorizer.fit_transform(preprocess_reviews(train["review"]))

    print("Running sentiment analysis model (~2 mins)...")
    # Optimal regularization parameter setting for ridge regression
    best_C = 21.54434690031882
    model = LogisticRegression(C=best_C, penalty="l2", max_iter=100000, random_state=SEED)  
    train_X = dtm_vocab_train
    train_y = train["sentiment"]
    model.fit(train_X, train_y)
    
    test_X = top_feature_vectorizer.transform(preprocess_reviews(test["review"]))
    
    pred_y = model.predict_proba(test_X)[:, 1]  # Predict probabilities for class 1 (positive review)
    
    output = pd.DataFrame()
    output['id'] = test['id']
    output['prob'] = pred_y
    output.to_csv('mysubmission.csv', index=False)
    print("Done")



if __name__ == '__main__':
    main()
