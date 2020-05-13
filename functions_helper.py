import pandas as pd
import math
import copy
import numpy as np
import itertools
import more_itertools as mit
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import re
import time

################################################################################
## DATA preprocessing
################################################################################

def preprocess_corpus(data):
    ps = PorterStemmer()

    def stemSentence(sentence, ps):
        token_words = word_tokenize(sentence)
        stem_sentence = []
        for word in token_words:
            stem_sentence.append(ps.stem(word))
            stem_sentence.append(" ")
        return "".join(stem_sentence)

    data['TEXT'] = data.apply(lambda x: stemSentence(x['TEXT'], ps), axis=1)

    return data


def preprocess_queries(corpus, queries, output_string=False):
    def remove_punctuations(text):  # remove punctuation
        for punctuation in string.punctuation:
            text = text.replace(punctuation, '')
        return text

    def remove_numbers(text):  # remove numbers
        return re.sub('[0-9]+', '', text)

    def lower_case(text):  # lower case
        text = text.lower()
        return text

    def tokenize(text):  # tokenize
        return word_tokenize(text)

    stop = set(stopwords.words('english'))

    def stop_words(tokens):  # stop words
        filtered_words = []
        for word in tokens:
            if word not in stop:
                filtered_words.append(word)
        return filtered_words

    ps = PorterStemmer()

    def stemming(tokens, ps):  # stemming
        return [ps.stem(w) for w in tokens]

    def corpus_vocab(corpus):
        vocab = []
        corpus_tokens = corpus.apply(lambda x: word_tokenize(x['TEXT']), axis=1)
        for i, j in corpus_tokens.iteritems():
            for token in j:
                if token not in vocab:
                    vocab.append(token)
        return vocab

    v = corpus_vocab(corpus)

    def filter_query(tokens):
        t = []
        for token in tokens:
            if token in v:
                t.append(token)
        return t

    def create_string(tokens):
        s = []
        for token in tokens:
            s.append(token)
            s.append(" ")
        return "".join(s)

    # apply functions
    queries['TEXT'] = queries.apply(lambda x: remove_punctuations(x['TEXT']), axis=1)
    queries['TEXT'] = queries.apply(lambda x: remove_numbers(x['TEXT']), axis=1)
    queries['TEXT'] = queries.apply(lambda x: lower_case(x['TEXT']), axis=1)
    queries['TEXT'] = queries.apply(lambda x: tokenize(x['TEXT']), axis=1)
    queries['TEXT'] = queries.apply(lambda x: stop_words(x['TEXT']), axis=1)
    queries['TEXT'] = queries.apply(lambda x: stemming(x['TEXT'], ps), axis=1)
    queries['TEXT'] = queries.apply(lambda x: filter_query(x['TEXT']), axis=1)
    if output_string:
        queries['TEXT'] = queries.apply(lambda x: create_string(x['TEXT']),
                                        axis=1)  # we need a string because pandas.Datarame.isin
    # function is not working with unhashable objects

    return queries
################################################################################
## TF-IDF calculations
################################################################################

# Term frequency
def tf(corpus, column_name):
    def tokenize(string):
        return string.split()

    def tf_string(string):
        # create bag of words from the string
        bow = tokenize(string)

        tf_dict = {}
        for word in bow:
            if word in tf_dict:
                tf_dict[word] += 1
            else:
                tf_dict[word] = 1

        for word in tf_dict:
            tf_dict[word] = 1 + math.log(tf_dict[word])

        return tf_dict

    # call our function on every doc and store all these tf dictionaries.
    tf_dict = {}
    for index, row in corpus.iterrows():
        doc_dict = tf_string(row[column_name])
        tf_dict[index] = doc_dict

    return tf_dict


# Inversed document frequency
def idf(corpus, tf_dict):
    # nomber of documents in corpus
    no_of_docs = len(corpus.index)

    # term - key, number of docs term occured in
    def count_occurances(tf_dict):
        count_dict = {}
        for key in tf_dict:
            for key in tf_dict[key]:
                if key in count_dict:
                    count_dict[key] += 1
                else:
                    count_dict[key] = 1
        return count_dict

    idf_dict = {}

    count_dict = count_occurances(tf_dict)

    for key in count_dict:
        idf_dict[key] = math.log(no_of_docs / count_dict[key])

    return idf_dict

# TF-IDF
def tf_idf(tf_dict, idf_dict):
    tf_idf_dict = copy.deepcopy(tf_dict)
    for doc, value in tf_idf_dict.items():
        for word, value in tf_idf_dict[doc].items():
            tf_idf_dict[doc][word] = value * idf_dict[word]
    return tf_idf_dict


# Convert tf_idf_dict to matrix
def tf_idf_to_matrix(tf_idf_dict):
    tf_idf_matrix = pd.DataFrame.from_dict(tf_idf_dict,
                                           orient = 'index').fillna(0) # if word does not appear in doc we change NaN to
    return tf_idf_matrix.sort_index()

################################################################################
## Cosine similarity
################################################################################

# Cosine similarity
def cosine_similarity(v1, v2):
    def vector_magnitude(v):
        return np.linalg.norm(v)

    def dot_product(v1, v2):
        return np.dot(v1, v2)

    return dot_product(v1, v2) / (vector_magnitude(v1) * vector_magnitude(v2))

################################################################################
## Evaluation metrics
################################################################################


def apk(actual, predicted):
    """
    Computes the average precision at k.
    This function computes the average prescision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), len(predicted))

def mapk(actual, predicted, k=5):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a,p,k) for a,p in zip(actual, predicted)])

def dcg(element_list):
    """
    Discounted Cumulative Gain (DCG)
    Parameters:
        element_list - a list of ranks Ex: [5,4,2,2,1]
    Returns:
        score
    """
    score = 0.0
    for order, rank in enumerate(element_list):
        score += float(rank)/math.log((order+2))
    return score


def ndcg(reference, hypothesis):
    """
    Normalized Discounted Cumulative Gain (nDCG)
    Normalized version of DCG:
        nDCG = DCG(hypothesis)/DCG(reference)
    Parameters:
        reference   - a gold standard (perfect) ordering Ex: [5,4,3,2,1]
        hypothesis  - a proposed ordering Ex: [5,2,2,3,1]
    Returns:
        ndcg_score  - normalized score
    """
    if dcg(reference) == 0:
        return 0
    else:
        return dcg(hypothesis)/dcg(reference)

################################################################################
## Query retrieval
################################################################################
#vectorize a single query into a vector from string or list of tokens
def vectorize_query(textstring):
    if type(textstring) == str:
        tokenized_query = textstring.split()
    else:
        tokenized_query = textstring

    df_query = tf_idf_matrix[0:0]  # dataframe of tf-idf weights of a query
    df_query = df_query.append(pd.Series(0, index=df_query.columns), ignore_index=True)
    for token in tokenized_query:
        for col in df_query.columns:
            if token == col:
                df_query[col][0] = df_query[col][0] + 1  # raw term frequency

    df_query = df_query.replace(0, np.nan)

    df_query = np.log(df_query) + 1  # log term freq(as in the slides)

    df_query = df_query.fillna(0)

    for col in df_query.columns:
        df_query[col][0] = df_query[col][0] * idf_dict[col]

    return df_query

#retrieve for this query relevant documents
def true_relevant_docs(string_query):
    query_row = (queries_text.loc[queries_text['TEXT'].isin([string_query])])
    query_id = query_row.iloc[0]["ID"]
    relevance_lvl = [1, 2]
    return queries_relevance.loc[queries_relevance['QUERY_ID'].isin([query_id]) & queries_relevance['RELEVANCE_LEVEL'].isin(relevance_lvl)]


# retrieve k documents with highest cousine similarity between projected query and doc vectors
def retrieve(position, queries, docs, k=10, random_projections = False):
    df = corpus.copy()

    sim = []  # to store cosine similarities
    sort_sim = []  # sorted cosine similarities
    i = 0
    for doc in docs:
        #in case of random projections calculate dot product since our document and query vectors would be unit-normalized
        if random_projections:
            sim.append([i, np.dot(queries[position], doc)])
        else:
            sim.append([i, cosine_similarity(queries[position], doc)])
        i += 1
    sort_sim = sorted(sim, key=lambda cos: cos[1], reverse=True)
    ids = []

    for j in range(k):
        ids.append(sort_sim[j][0])

    return df.iloc[ids]


def evaluate_single_retrieve(position, queries, docs, k=10, random_projections = False):
    ## returns the triple (Precision, Average Precision, Normalized Discounted Cumulative Gain)
    relevant = true_relevant_docs(queries_text['TEXT'][position])
    retrieved_df = retrieve(position, queries, docs, k=k, random_projections=random_projections)
    ids_retrieved = []
    for i in range(len(retrieved_df)):
        ids_retrieved.append(retrieved_df.iloc[i].ID)
    ids_retrieved.sort()

    ids_true_relevant = []
    for i in range(len(relevant)):
        ids_true_relevant.append(relevant.iloc[i].DOC_ID)
    ids_true_relevant.sort()

    # count true positives and false positives
    tp = 0
    fp = 0
    for i in ids_retrieved:
        for j in ids_true_relevant:
            if i == j:
                tp += 1
                break
            else:
                if i < j:
                    fp += 1
                    break
                else:
                    continue
    if (tp == 0) & (fp == 0):
        precision = 0
    else:
        precision = tp / (tp + fp)
    # cannot calculate recall, since we predefined the number of retrieved documents => apriori algorithm cannot retrieve all documents

    # then calculate Average precision across retrieved documents
    ap = apk(ids_true_relevant, ids_retrieved)

    # since we have graded relevance annotations, we can also calculate Normalized Discounted Cumulative Gain
    list_of_ranks_of_retrieved_docs = []
    for i in ids_retrieved:
        if i in ids_true_relevant:
            list_of_ranks_of_retrieved_docs.append(relevant.loc[relevant['DOC_ID'].isin([i])].RELEVANCE_LEVEL.iloc[0])
        else:
            list_of_ranks_of_retrieved_docs.append(0)

    list_of_ranks_of_relevant_docs = []
    for i in ids_true_relevant:
        list_of_ranks_of_relevant_docs.append(relevant.loc[relevant['DOC_ID'].isin([i])].RELEVANCE_LEVEL.iloc[0])
    list_of_ranks_of_relevant_docs.sort(reverse=True)

    k = len(list_of_ranks_of_retrieved_docs)
    list_of_ranks_of_relevant_docs = list_of_ranks_of_relevant_docs[:k]

    return precision, ap, ndcg(list_of_ranks_of_relevant_docs, list_of_ranks_of_retrieved_docs)


def full_evaluation(queries, docs, k=10, random_projections = False):
    evaluation = queries_text.copy()
    evaluation.insert(2, "Precision", 0)
    evaluation.insert(3, "Average Precision", 0)
    evaluation.insert(4, "nDCG", 0)

    for i in range(len(evaluation)):
        p, a, n = evaluate_single_retrieve(i, queries, docs, k=k, random_projections=random_projections)
        evaluation.loc[i, 'Precision'] = p
        evaluation.loc[i, 'Average Precision'] = a
        evaluation.loc[i, 'nDCG'] = n

    print('Average precision across all queries = ' + str(evaluation['Precision'].mean()))
    print('Mean Average Precision = ' + str(evaluation['Average Precision'].mean()))
    print('Average nDCG = ' + str(evaluation['nDCG'].mean()))

    return evaluation

################################################################################
## Random Projections
################################################################################

def norm(vectors):
    norm_vectors = []
    if len(vectors) == 1:
        norm_vectors.append(vectors/np.linalg.norm(vectors))
    else:
        for vec in vectors:
            norm_vectors.append(vec/np.linalg.norm(vec))
    return np.array(norm_vectors)

def get_random_vectors(dim,m):
    vectors = np.random.randn(m, dim)
    return norm(vectors)

def compute_hash(docs, rnd_vec, t):
    hashed_doc_vectors = []
    #for each document in document collection
    for doc in docs:
        hashed_dot_product = []
        inner_product = doc.dot(rnd_vec.transpose())
        for i in inner_product:
            if i>t:
                hashed_dot_product.append(1)
            else:
                hashed_dot_product.append(0)
        hashed_doc_vectors.append(hashed_dot_product)
    return np.array(hashed_doc_vectors)