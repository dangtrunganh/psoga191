import os
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import numpy as np
from nltk.stem import PorterStemmer
import math



def list_sentences_to_string(list_sentences):
    return " ".join(list_sentences)


def weight(article, list_sentences, word_in_sents,  return_voca=False):
    article_text = list_sentences_to_string(list_sentences)

    # word_in_sents = word_appear_sent(article)


    word_frequencies = {}
    for word in nltk.word_tokenize(article_text):
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1
    if return_voca:
        voca = list(word_frequencies)
    else:
        voca = None
    if len(word_frequencies) == 0:
        maximum_prequency = 0
    else:
        maximum_prequency = max(word_frequencies.values())

    for word in word_frequencies.keys():
        if maximum_prequency == 0:
            word_frequencies[word] = 0
        else:
            word_frequencies[word] = (word_frequencies[word]/maximum_prequency)*word_in_sents[word]
    return word_frequencies, voca



def word_appear_sent(article):
    article_text = list_sentences_to_string(article)

    word_frequencies = {}
    for word in nltk.word_tokenize(article_text):
        if word not in word_frequencies.keys():
            word_frequencies[word] = 1
        else:
            word_frequencies[word] += 1

    word_in_sent = {}
    for word in word_frequencies.keys():
        word_in_sent[word] = 0
        for sentence in article:
            if word in nltk.word_tokenize(sentence):
                word_in_sent[word] += 1
    value_for_word_in_sent = {}
    n = len(article)
    for word in word_frequencies.keys():
        value_for_word_in_sent[word] = math.log(n/(1+word_in_sent[word]))
    return value_for_word_in_sent




def normalize(vec):
    return vec / np.sqrt(np.sum(vec ** 2))

    
def simCos(vec1, vec2):
    norm_vec1 = normalize(vec1)
    norm_vec2 = normalize(vec2)
    return np.sum(norm_vec1 * norm_vec2)

def get_vec_weight(S, voca): #S là dict của 1 câu, cần chuyển kích thước bằng vs dict của 1 document
    vec = np.zeros(len(voca))
    for i in range(len(voca)):
        try:
            vec[i] = S[voca[i]]
            
        except:
            continue
    return vec
def sim_2_sent(sentences):    
    word_in_sents = word_appear_sent(sentences)
    d_dict, voca = weight(sentences, sentences,  word_in_sents, return_voca=True)
    sim2sents = []
    for i in range(len(sentences)):
        sim2sents.append([])
        for j in range(len(sentences)):
            sent1_dict, _ = weight(sentences, [sentences[i]], word_in_sents)
            sent1_vec = get_vec_weight(sent1_dict, voca)
            sent2_dict, _ = weight(sentences, [sentences[j]], word_in_sents )
            sent2_vec = get_vec_weight(sent2_dict, voca)
            sim2sents[i].append(simCos(sent1_vec, sent2_vec))
    return sim2sents

def sim_with_title(sentences, title):
    word_in_sents = word_appear_sent(sentences)
    d_dict, voca = weight(sentences, sentences,word_in_sents, return_voca=True)
    title_dict, _ = weight(sentences, [title], word_in_sents)
    title_vec = get_vec_weight(title_dict, voca)
    simWithTitle = []
    for sent in sentences:
        s_i_dict, _ = weight(sentences, [sent], word_in_sents)
        s_i = get_vec_weight(s_i_dict, voca)
        simT = simCos(s_i, title_vec)
        simWithTitle.append(simT)
    return simWithTitle

def sim_with_doc(sentence, sentences):
    word_in_sents = word_appear_sent(sentences)
    d_dict, voca = weight(sentences, sentences,word_in_sents, return_voca=True)
    document_ = get_vec_weight(d_dict, voca)
    sentence_dict, _ = weight(sentences, [sentence], word_in_sents)
    sentence_ = get_vec_weight(sentence_dict, voca)
    return simCos(sentence_, document_)



def count_noun(sentences): #đếm số danh từ 
    number_of_nouns = []
    for sentence in sentences:
        text = nltk.word_tokenize(sentence)
        post = nltk.pos_tag(text)
        # noun_list = ['NN', 'NNP', 'NNS', 'NNPS']
        noun_list = ['NNP']
        num = 0
        for k,v in post:
            if v in noun_list:
                num += 1
        number_of_nouns.append(num)
    return number_of_nouns


def preprocess_raw_sent(raw_sent):
    words = nltk.word_tokenize(raw_sent)
    preprocess_words = ""
    stopwords = nltk.corpus.stopwords.words('english')
    # stemmer= PorterStemmer()
    for word in words:
        if word.isalpha():
            if word not in stopwords:
                word = word.lower()
                # word = stemmer.stem(word)
                word = " " + word
                preprocess_words += word
    preprocess_words = preprocess_words.strip()
    return preprocess_words

def preprocess_numberOfNNP(raw_sent):
    words = nltk.word_tokenize(raw_sent)
    preprocess_words = ""
    stopwords = nltk.corpus.stopwords.words('english')
    # stemmer= PorterStemmer()
    for word in words:
        if word.isalpha():
            if word not in stopwords:
                word = " " + word
                preprocess_words += word
    preprocess_words = preprocess_words.strip()
    return preprocess_words

def preprocess_for_article(raw_sent):
    words = nltk.word_tokenize(raw_sent)
    preprocess_words = ""
    # stopwords = nltk.corpus.stopwords.words('english')
    # stemmer= PorterStemmer()
    for word in words:
        # if word.isalpha():
            # if word not in stopwords:
                # word = word.lower()
                # word = stemmer.stem(word)
                word = " " + word
                preprocess_words += word
    preprocess_words = preprocess_words.strip()
    return preprocess_words