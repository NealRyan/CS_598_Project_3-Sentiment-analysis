import argparse
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV


SEED = 4031
np.random.seed(SEED)

def main():
    parser = argparse.ArgumentParser(description="Process three input files.")

    parser.add_argument('myvocab_file', type=str, help='Path to myvocab.txt')
    parser.add_argument('train_file', type=str, help='Path to train.tsv')
    parser.add_argument('test_file', type=str, help='Path to test.tsv')

    args = parser.parse_args()

    vocab = args.myvocab_file
    train = args.train_file
    test = args.test_file

    train = pd.read_csv(train, sep='\t')
    test = pd.read_csv(test, sep='\t')
    print(type(train))

    stopwords = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "their", "they", "his", \
             "her", "she", "he", "a", "an", "and", "is", "was", "are", "were", "him", "himself", "has", "have", "it", "its", \
             "the", "us"]

    vocab = expand_contractions(vocab)

    vectorizer = CountVectorizer(
        preprocessor=lambda x: x.lower(), # Convert to lowercase
        vocabulary = vocab,
        stop_words=stopwords,             # Remove stop words
        ngram_range=(1, 4),               # Use 1- to 4-grams
        min_df=0.001,                     # Minimum term frequency
        max_df=0.5,                       # Maximum document frequency
        token_pattern=r"\b[\w+\|']+\b"    # Use word tokenizer, but don't split on apostrophes
    )

    dtm_train = vectorizer.fit_transform(train["review"])
    feature_ngrams = vectorizer.get_feature_names_out()

    dtm_pos = dtm_train[train.sentiment == 1, :]
    dtm_neg = dtm_train[train.sentiment == 0, :]

    dtm_pos_count = dtm_pos.shape[0]
    dtm_neg_count = dtm_neg.shape[0]
    dtm_pos_count, dtm_neg_count

    dtm_pos_means = np.empty(feature_ngrams.shape[0])
    dtm_pos_vars = np.empty(feature_ngrams.shape[0])

    dtm_neg_means = np.empty(feature_ngrams.shape[0])
    dtm_neg_vars = np.empty(feature_ngrams.shape[0])

    for col in range(feature_ngrams.shape[0]):
        pos_col_array = dtm_pos[:, col].toarray()
        dtm_pos_means[col] = np.mean(pos_col_array)
        dtm_pos_vars[col] = np.var(pos_col_array, ddof=1)
        
        neg_col_array = dtm_neg[:, col].toarray()
        dtm_neg_means[col] = np.mean(neg_col_array)
        dtm_neg_vars[col] = np.var(neg_col_array, ddof=1)

    t_statistics = (dtm_pos_means - dtm_neg_means) / np.sqrt((dtm_pos_vars/dtm_pos_count) + (dtm_neg_vars/dtm_neg_count))

    feature_statistic_df = pd.DataFrame({"feature": feature_ngrams.tolist(), "statistic": t_statistics.tolist()})

    n_terms = 2000

    feature_statistic_df["abs_statistic"] = abs(feature_statistic_df["statistic"])

    top_features = feature_statistic_df.sort_values(by="abs_statistic", ascending=False).iloc[:n_terms, 0]


    only_positive = feature_ngrams[np.logical_and((dtm_pos_means > 0), (dtm_neg_means == 0))]
    only_negative = feature_ngrams[np.logical_and((dtm_pos_means == 0), (dtm_neg_means > 0))]
    top_features_list = list(set(top_features.tolist() + only_positive.tolist() + only_negative.tolist()))
    top_features_df = feature_statistic_df[feature_statistic_df['feature'].isin(top_features_list)]
    top_features_df = top_features_df.sort_values(by='abs_statistic', ascending=False)

    custom_vectorizer = CountVectorizer(
        vocabulary=top_features_df['feature'],          # The top 2000 features
        stop_words=stopwords,             # Remove stop words
        ngram_range=(1, 4),               # Use 1- to 4-grams
        min_df=0.001,                     # Minimum term frequency
        max_df=0.5,                       # Maximum document frequency
        token_pattern=r"\b[\w+\|']+\b"    # Use word tokenizer, but don't split on apostrophes
    )

    X_train = custom_vectorizer.fit_transform(preprocess_reviews(train['review']))
    Y_train = train['sentiment']

    best_tokens = find_best_tokens(1000, 0.04604, X_train, Y_train)

    top_features_df = pd.DataFrame(top_features.items(), columns=['token', 'feature'])
    best_tokens_df = pd.DataFrame(best_tokens, columns=['index', 'value']).set_index('index')


    lasso_best_tokens_df = top_features_df.join(best_tokens_df)
    lasso_best_tokens_df = lasso_best_tokens_df.dropna()
    lasso_best_tokens_df['weight'] = lasso_best_tokens_df['value'].abs()

    lasso_best_tokens_df = lasso_best_tokens_df.sort_values(by='weight', ascending=False)
    top_features = lasso_best_tokens_df['feature'].tolist()

    top_feature_vectorizer = CountVectorizer(
    vocabulary=top_features,          # The top 200 features
    stop_words=stopwords,             # Remove stop words
    ngram_range=(1, 4),               # Use 1- to 4-grams
    min_df=0.001,                     # Minimum term frequency
    max_df=0.5,                       # Maximum document frequency
    token_pattern=r"\b[\w+\|']+\b"    # Use word tokenizer, but don't split on apostrophes
)

    dtm_vocab_train = top_feature_vectorizer.fit_transform(train["review"])
    grid_search = LogisticRegressionCV(Cs=10, cv=5, penalty="l2", scoring="roc_auc", max_iter=100000, random_state=SEED, verbose=1)
    all_train_y = train["sentiment"]
    grid_search.fit(dtm_vocab_train, all_train_y)
    best_C = grid_search.C_[0]



    model = LogisticRegression(C=best_C, penalty="l2", max_iter=100000, random_state=SEED, verbose=1)  
    train_X = top_feature_vectorizer.fit_transform(preprocess_reviews(train["review"]))
    train_y = train["sentiment"]
    model.fit(train_X, train_y)
    
    test_X = top_feature_vectorizer.transform(preprocess_reviews(test["review"]))
    
    pred_y = model.predict_proba(test_X)[:, 1]  # Predict probabilities for class 1 (positive review)
    
    output = pd.DataFrame()
    output['id'] = train['id']
    output['prob'] = pred_y
    output.to_csv('output_file.csv', index=False)
    print("Done")

def preprocess_reviews(reviews):
    reviews = reviews.str.replace('<.*?>', ' ', regex=True)
    reviews = reviews.str.lower()
    reviews = expand_contractions(reviews)
    return reviews

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
        if isinstance(reviews, str):
                reviews = pd.Series([reviews])

        reviews = reviews.str.replace(contraction, contractions[contraction], regex=True)
        
    return reviews

def find_best_tokens(num, c, X_train, Y_train):
    for i in range(1000):
        lasso_log_model = LogisticRegression(C=c, penalty='l1', solver='liblinear', max_iter=100000)  # very high max iter to ensure converge
        #X_train = custom_vectorizer.fit_transform(preprocess_reviews(all_train['review']))
        #Y_train = all_train['sentiment']
        lasso_log_model.fit(X_train, Y_train)

        best_tokens = [[i, coef] for i, coef in enumerate(lasso_log_model.coef_[0]) if coef != 0]

        num_tokens = len(best_tokens)
        print(f'number of tokens: {num_tokens}')
        print(f'old c: {c}')
        
        diff = num_tokens-num

        if num_tokens == num:
            return best_tokens
        elif num_tokens > num:
            c = c*.999
        elif num_tokens < num:
            c = c*1.001

        print(f'new c: {c}')
    print("Bad initial c value, try another value")
    raise Exception


if __name__ == '__main__':
    main()

