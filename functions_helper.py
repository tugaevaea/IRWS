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
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import collections

import nltk
nltk.download('punkt')
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


def preprocess_queries(corpus, queries, output_string = False):
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

#create tf-idf matrix for queries
def queries_tf_idf(tf_idf_matrix, idf_dict, queries, tokenize = True):
    #create tf-idf matrix of queries
    tf_idf_queries = tf_idf_matrix[0:0]
    
    for i in range(len(queries)):
        tf_idf_queries = tf_idf_queries.append(pd.Series(0, index=tf_idf_queries.columns), ignore_index=True)
        #count occurances
        if tokenize:
            for token in (queries['TEXT'][i]):
                for col in tf_idf_queries.columns:
                    if token == col:
                        tf_idf_queries[col][i] = tf_idf_queries[col][i] + 1
        else: 
            for token in (queries['TEXT'][i]).split():
                for col in tf_idf_queries.columns:
                    if token == col:
                        tf_idf_queries[col][i] = tf_idf_queries[col][i] + 1
                
    #calculate log tf
    tf_idf_queries = np.log(tf_idf_queries) + 1 
    tf_idf_queries = tf_idf_queries.replace(-np.inf,0)
    
    for i in range(len(queries)):
        for col in tf_idf_queries.columns:
            tf_idf_queries[col][i] = tf_idf_queries[col][i] * idf_dict[col]
    
    return tf_idf_queries

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
## Tiered index
################################################################################


################################################################################
## Pre-clustering
################################################################################

def allocate_docs_to_clusters(rs, sqrt_n, cosine = False, Faiss = False):
    
    def set_leaders(random_state = rs):
        #Set number of clusters at initialisation time
        #sqrt_n = round(math.sqrt(no_of_docs))
        #we randomly select sqrt(N) documents from the corpus, which we call leaders
        leaders = tf_idf_matrix.sample(sqrt_n, random_state = rs)
        leaders = leaders.sort_index()
        return leaders

    leaders = set_leaders(random_state = rs)
    
    # For every other document in the collection
    # 1. Compute the similarities (cosine of the angle between TF-IDF vectors) with all leaders
    # 2. Add the document to the cluster of the most similar leader
    if (cosine and Faiss) or ((not cosine) and (not Faiss)):
        print('both true')
        return [], []
    
    
    elif cosine == True:
        cluster_list = []

        for i in range(sqrt_n):
            cluster_list.append([])

        for i in range(no_of_docs):
            cosines = []
            for j in leaders.index:
                cosines.append(cosine_similarity(tf_idf_matrix.loc[i],leaders.loc[j]))
            m = max(cosines)
            index_of_max = [l for l, b in enumerate(cosines) if b == m]
            cluster_list[index_of_max[0]].append(i) #if there are two equal max values of cosine similarity use the smaller index by default
        return leaders, cluster_list
    
    elif Faiss == True:
        
        leaders = leaders.astype('float32')
        index = faiss.IndexFlatL2(len(leaders.columns))
        index.add(np.ascontiguousarray(leaders.values))
        
        cluster_list = []

        for i in range(sqrt_n):
            cluster_list.append([])

        for i in range(no_of_docs):
            doc = np.array([tf_idf_matrix.loc[i]])
            D, I = index.search(doc, 1)
            cluster_list[I[0][0]].append(i)
        
        return leaders, cluster_list
    
    else:
        print('exeption')
        return [],[]

###########################
## Basic pre-clustering
###########################

#construct function, which uses query q(should be already in the vector form) as input, required similarity of the doc to be retrieved - threshold, and
#necessary number of documents to be retrieved - K (5 most similar docs in the cluster by default)

def ir_preclustering(q, threshold = 0, K = 5): 
    
    sim_to_leaders = [] #array of cosine similarities of q to leaders
    retrieved_docs = [] #array of the most similar docs to be returned by the function
    
    for i in range(len(leaders.index)):
        sim_to_leaders.append(cosine_similarity(q,leaders.iloc[i]))
        m = max(sim_to_leaders)
        index_of_max = [l for l, b in enumerate(sim_to_leaders) if b == m] #odinal number of most similar leader => use this cluster
    
    sim_to_docs = [] #array of cosine similarities of q to all docs in the chosen cluster
    for doc in cluster_list[index_of_max[0]]:
        sim_to_docs.append(cosine_similarity(q,tf_idf_matrix.iloc[doc]))
        
    ins = np.argsort(sim_to_docs) #returns the indices that would sort an array of similarities to docs in ascending order
    ins = ins[::-1] #but we need descending (most similar in the beginning of the list)
    
    if threshold == 0: #proceed only with K
        if len(ins)>=K:
            for k in range(K):
                retrieved_docs.append(cluster_list[index_of_max[0]][ins[k]])
        else:
            K=len(ins)
            for k in range(K):
                retrieved_docs.append(cluster_list[index_of_max[0]][ins[k]])

        
    else:
        if sim_to_docs[ins[0]] < threshold:
            print('no documents satisfy necessary level of threshold similarity')
            return None
        
        for sim in sim_to_docs:
            if sim >= threshold:
                retrieved_docs.append(cluster_list[index_of_max[0]][sim_to_docs.index(sim)])
            if len(retrieved_docs) < K:
                print('number of documents that satisfy threshold similarity is less than required \(less than K\)')
    
    return corpus.iloc[retrieved_docs]

def retrieve_with_preclustering(string_query, k = 5, IDs_of_retrieved_docs = False):
    vector_q = vectorize_query(string_query)
    return ir_preclustering(vector_q.iloc[0], K = k)

def evaluate_retrieve_with_preclustering(position):
    
    ## takes ordinal number of a query as an input 
    ## returns the triple (Precision, Average Precision, Normalized Discounted Cumulative Gain)
    
    retrieved_df = ir_preclustering(tf_idf_queries.iloc[position])
    ids_retrieved = []
    for i in range(len(retrieved_df)):
        ids_retrieved.append(retrieved_df.iloc[i].ID)
    ids_retrieved.sort()
    
    relevant = true_relevant_docs(queries_text['TEXT'][position])
    ids_true_relevant = []
    for i in range(len(relevant)):
        ids_true_relevant.append(relevant.iloc[i].DOC_ID)
    ids_true_relevant.sort()
    
    #count true positives and false positives
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
        precision = tp/(tp+fp)
    #cannot calculate recall, since we predefined the number of retrieved documents => apriori algorithm cannot retrieve all documents
    
    #then calculate Average precision across retrieved documents
    ap = apk(ids_true_relevant, ids_retrieved)
    
    #since we have graded relevance annotations, we can also calculate Normalized Discounted Cumulative Gain
    list_of_ranks_of_retrieved_docs = []
    for i in ids_retrieved:
        if i in ids_true_relevant:
            list_of_ranks_of_retrieved_docs.append(relevant.loc[relevant['DOC_ID'].isin([i])].RELEVANCE_LEVEL.iloc[0])
        else:
            list_of_ranks_of_retrieved_docs.append(0)

                                               
    list_of_ranks_of_relevant_docs = []
    for i in ids_true_relevant:
        list_of_ranks_of_relevant_docs.append(relevant.loc[relevant['DOC_ID'].isin([i])].RELEVANCE_LEVEL.iloc[0])
    list_of_ranks_of_relevant_docs.sort(reverse = True)
    
    k=len(list_of_ranks_of_retrieved_docs)
    list_of_ranks_of_relevant_docs = list_of_ranks_of_relevant_docs[:k]
        
    return precision, ap, ndcg(list_of_ranks_of_relevant_docs, list_of_ranks_of_retrieved_docs)       
                
    

    

###########################
## Faiss pre-clustering
###########################
def set_indeces_for_faiss():
    index = faiss.IndexFlatL2(len(leaders.columns))
    index.add(np.ascontiguousarray(leaders.values))
    
    indices = []
    for i in range(len(leaders)):
        index2 = faiss.IndexFlatL2(len(leaders.columns))
        index2.add(np.ascontiguousarray(tf_idf_matrix.loc[cluster_list[i]]))
        indices.append(index2)
        
    return index, indices


#find the nearest leader
def ir_preclustering_faiss(q, K = 5, threshold = 0):
    
    

    q = np.array([q])
    D, I = index.search(q, 1) #returning distance and index of the nearest leader
    
    index2 = indices[I[0][0]]
    
    if threshold == 0: #proceed only with K
        
        if len(cluster_list[I[0][0]]) < K:
            print('asked number of documents to be retrieved is larger than the number of documents in the cluster; \nall documents in the cluster are retrieved')
            return corpus.iloc[cluster_list[I[0][0]]]   
        else:
            DD, II = index2.search(q, K) #returning distances and indexes of the nearest documents in the cluster (sorted by distance)
            return corpus.iloc[II[0]]
            
        
    else:
        DD, II = index2.search(q, K)
        DD = [1 - x for x in DD] #now DD are not distances, but similarities
        
        if DD[0] < threshold:
            return None
        
        for sim in DD:
            if sim < threshold:
                DD.pop(sim)
        
        if len(DD) < K:
                print('number of documents that satisfy threshold similarity is less than required \(less than K\)')
       
        return corpus.iloc[II[0]]
    
    
def retrieve_with_preclustering_faiss(string_query, k = 5, IDs_of_retrieved_docs = False):
    vector_q = vectorize_query(string_query)
    return ir_preclustering_faiss(vector_q.iloc[0].astype('float32'), K = k)

def evaluate_retrieve_with_preclustering_faiss(position):
    ## returns the triple (Precision, Average Precision, Normalized Discounted Cumulative Gain)
    
    retrieved_df = ir_preclustering_faiss(tf_idf_queries.iloc[position].astype('float32'))
    ids_retrieved = []
    
    for i in range(len(retrieved_df)):
        ids_retrieved.append(retrieved_df.iloc[i]['ID'])
    ids_retrieved.sort()
    
    relevant = true_relevant_docs(queries_text['TEXT'][position])
    ids_true_relevant = []
    for i in range(len(relevant)):
        ids_true_relevant.append(relevant.iloc[i].DOC_ID)
    ids_true_relevant.sort()
    
    #count true positives and false positives
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
        precision = tp/(tp+fp)
    #cannot calculate recall, since we predefined the number of retrieved documents => apriori algorithm cannot retrieve all documents
    
    #then calculate Average precision across retrieved documents
    ap = apk(ids_true_relevant, ids_retrieved)
    
    #since we have graded relevance annotations, we can also calculate Normalized Discounted Cumulative Gain
    list_of_ranks_of_retrieved_docs = []
    for i in ids_retrieved:
        if i in ids_true_relevant:
            list_of_ranks_of_retrieved_docs.append(relevant.loc[relevant['DOC_ID'].isin([i])].RELEVANCE_LEVEL.iloc[0])
        else:
            list_of_ranks_of_retrieved_docs.append(0)

                                               
    list_of_ranks_of_relevant_docs = []
    for i in ids_true_relevant:
        list_of_ranks_of_relevant_docs.append(relevant.loc[relevant['DOC_ID'].isin([i])].RELEVANCE_LEVEL.iloc[0])
    list_of_ranks_of_relevant_docs.sort(reverse = True)
    
    k=len(list_of_ranks_of_retrieved_docs)
    list_of_ranks_of_relevant_docs = list_of_ranks_of_relevant_docs[:k]
        
    return precision, ap, ndcg(list_of_ranks_of_relevant_docs, list_of_ranks_of_retrieved_docs) 
    
    
###########################
## KMeans pre-clustering
###########################

#construct function, which uses query q(should be already in the vector form) as input, required similarity of the doc to be retrieved - threshold, and
#necessary number of documents to be retrieved - K (5 most similar docs in the cluster by default)

def ir_preclustering_kmeans(q, threshold = 0, K = 5):
#     q = [q]
    sim_to_centers = [] #array of cosine similarities of q to centers
    retrieved_docs = [] #array of the most similar docs to be returned by the function
    
    for i in range(len(centers)):
        sim_to_centers.append(cosine_similarity(q,centers[i]))
    m = max(sim_to_centers)
    index_of_max = [l for l, b in enumerate(sim_to_centers) if b == m] #odinal number of most similar leader => use this cluster
    
    sim_to_docs = [] #array of cosine similarities of q to all docs in the chosen cluster
    for doc in cluster_list_kmeans[index_of_max[0]]:
        sim_to_docs.append(cosine_similarity(q,tf_idf_matrix.iloc[doc]))
        
    ins = np.argsort(sim_to_docs) #returns the indices that would sort an array of similarities to docs in ascending order
    ins = ins[::-1] #but we need descending (most similar in the beginning of the list)
    
    if threshold == 0: #proceed only with K
        if len(ins)>=K:
            for k in range(K):
                retrieved_docs.append(cluster_list_kmeans[index_of_max[0]][ins[k]])
        else:
            K=len(ins)
            for k in range(K):
                retrieved_docs.append(cluster_list_kmeans[index_of_max[0]][ins[k]])

        
    else:
        if sim_to_docs[ins[0]] < threshold:
            print('no documents satisfy necessary level of threshold similarity')
            return None
        
        for sim in sim_to_docs:
            if sim >= threshold:
                retrieved_docs.append(cluster_list_kmeans[index_of_max[0]][sim_to_docs.index(sim)])
            if len(retrieved_docs) < K:
                print('number of documents that satisfy threshold similarity is less than required \(less than K\)')
    
    return corpus.iloc[retrieved_docs]

def retrieve_with_preclustering_kmeans(string_query, k = 5, IDs_of_retrieved_docs = False):
    vector_q = vectorize_query(string_query)
    return ir_preclustering_kmeans(vector_q.iloc[0].astype('float32'), K = k)

def evaluate_retrieve_with_preclustering_kmeans(position):
    ## returns the triple (Precision, Average Precision, Normalized Discounted Cumulative Gain)
    
    retrieved_df = ir_preclustering_kmeans(tf_idf_queries.iloc[position].astype('float32'))
    ids_retrieved = []
    for i in range(len(retrieved_df)):
        ids_retrieved.append(retrieved_df.iloc[i].ID)
    ids_retrieved.sort()
    
    relevant = true_relevant_docs(queries_text['TEXT'][position])
    ids_true_relevant = []
    for i in range(len(relevant)):
        ids_true_relevant.append(relevant.iloc[i].DOC_ID)
    ids_true_relevant.sort()
    
    #count true positives and false positives
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
        precision = tp/(tp+fp)
    #cannot calculate recall, since we predefined the number of retrieved documents => apriori algorithm cannot retrieve all documents
    
    #then calculate Average precision across retrieved documents
    ap = apk(ids_true_relevant, ids_retrieved)
    
    #since we have graded relevance annotations, we can also calculate Normalized Discounted Cumulative Gain
    list_of_ranks_of_retrieved_docs = []
    for i in ids_retrieved:
        if i in ids_true_relevant:
            list_of_ranks_of_retrieved_docs.append(relevant.loc[relevant['DOC_ID'].isin([i])].RELEVANCE_LEVEL.iloc[0])
        else:
            list_of_ranks_of_retrieved_docs.append(0)

                                               
    list_of_ranks_of_relevant_docs = []
    for i in ids_true_relevant:
        list_of_ranks_of_relevant_docs.append(relevant.loc[relevant['DOC_ID'].isin([i])].RELEVANCE_LEVEL.iloc[0])
    list_of_ranks_of_relevant_docs.sort(reverse = True)
    
    k=len(list_of_ranks_of_retrieved_docs)
    list_of_ranks_of_relevant_docs = list_of_ranks_of_relevant_docs[:k]
        
    return precision, ap, ndcg(list_of_ranks_of_relevant_docs, list_of_ranks_of_retrieved_docs) 

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

def evaluate_preclustering():
    evaluation = queries_text.copy()
    evaluation.insert(2, "Precision", 0)
    evaluation.insert(3, "Average Precision", 0)
    evaluation.insert(4, "nDCG", 0)
    
    for i in range(len(evaluation)):

        p, a, n = evaluate_retrieve_with_preclustering(i,)
        evaluation.loc[i, 'Precision'] = p
        evaluation.loc[i, 'Average Precision'] = a
        evaluation.loc[i, 'nDCG'] = n
    
    print('Average precision across all queries = ' + str(evaluation['Precision'].mean()))
    print('Mean Average Precision = ' + str(evaluation['Average Precision'].mean()))
    print('Average nDCG = ' + str(evaluation['nDCG'].mean()))
    
    return evaluation

def evaluate_preclustering_faiss():
    evaluation = queries_text.copy()
    evaluation.insert(2, "Precision", 0)
    evaluation.insert(3, "Average Precision", 0)
    evaluation.insert(4, "nDCG", 0)
    
    for i in range(len(evaluation)):

        p, a, n = evaluate_retrieve_with_preclustering_faiss(i)
        evaluation.loc[i, 'Precision'] = p
        evaluation.loc[i, 'Average Precision'] = a
        evaluation.loc[i, 'nDCG'] = n
    
    print('Average precision across all queries = ' + str(evaluation['Precision'].mean()))
    print('Mean Average Precision = ' + str(evaluation['Average Precision'].mean()))
    print('Average nDCG = ' + str(evaluation['nDCG'].mean()))
    
    return evaluation


def evaluate_preclustering_kmeans():
    evaluation = queries_text.copy()
    evaluation.insert(2, "Precision", 0)
    evaluation.insert(3, "Average Precision", 0)
    evaluation.insert(4, "nDCG", 0)
    
    for i in range(len(evaluation)):

        p, a, n = evaluate_retrieve_with_preclustering_kmeans(i)
        evaluation.loc[i, 'Precision'] = p
        evaluation.loc[i, 'Average Precision'] = a
        evaluation.loc[i, 'nDCG'] = n
    
    print('Average precision across all queries = ' + str(evaluation['Precision'].mean()))
    print('Mean Average Precision = ' + str(evaluation['Average Precision'].mean()))
    print('Average nDCG = ' + str(evaluation['nDCG'].mean()))
    
    return evaluation

