import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import os
import pandas as pd
import fasttext
import numpy as np
from keras import regularizers, losses
from keras.models import Model, load_model

import numpy as np
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from geopy.geocoders import Nominatim
from collections import Counter


# ---------- Handle Text Data
def remove_punt(utterance):
    '''
        :param utterance: string -> textual utterance with no previews treatment
        :return: string with no punctuation
    '''
    if utterance == 'nan':
        return utterance

    removed_ponct = ''.join([w for w in utterance if w not in string.punctuation])
    return removed_ponct


def tokenize_text(dataset, column_name):
    '''
     :param dataset: dataframe -> columns of textual utterances with no previews treatment
     :param column_name: string
     :return: Series
     '''

    tokenizer = RegexpTokenizer(r'\w+')
    # remove punctuation and tokenize
    dataset[column_name] = dataset[column_name].apply(lambda x: tokenizer.tokenize(x.lower()))
    return dataset[column_name]


def remove_stopword(tokens):
    '''
    :param tokens: list of tokens
    :return: list of tokens
    '''

    stopList = stopwords.words('english')

    if type(tokens) == list:
        cleanTokens = [w for w in tokens if w not in stopList]
        return cleanTokens
    return tokens


def clean_stopwords(dt, col):
    '''
    :param serie: serie
    :return: Series
    '''

    dt[col] = dt[col].apply(lambda x: remove_stopword(x))
    return dt[col]


def token2prediction(tokens, model, fast):
    '''
    :param tokens: list of tokens
    :param model: keras machine learning model to do prediction
    :param fast: fasttext oject
    :return: list with fasttext word embeddings
    '''

    stopList = stopwords.words('english')
    lemmatizer = WordNetLemmatizer()
    lemTokens = []

    for t in tokens:
        # clean stop words
        if t in stopList:
            continue

        # lemmatize
        lemma = lemmatizer.lemmatize(t)

        # get word embedding
        emb = fast[lemma]

        # predict VAD - valence arousal dominance
        lemTokens.append(model.predict(np.array(emb).reshape(1, -1))[0])

    sentVAD = np.asarray(lemTokens)
    gobalVAD = np.sum(sentVAD, axis=0) / len(sentVAD)

    return gobalVAD.tolist()


def predict_sentiment(data, col, model, fast):
    '''
    :param data: dataframe
    :param col: string with column name
    :param model: keras machine learning model to do prediction
    :param fast: fasttext oject
    :return: Series with column treated
    '''

    data[col] = data[col].apply(lambda x: token2prediction(x, model, fast))
    return data[col]


def lemma2prediction(tokens, model, fast):
    '''
    :param tokens: list of tokens
    :param model: keras machine learning model to do prediction
    :param fast: fasttext oject
    :return: list with fasttext word embeddings
    '''

    lemTokens = []

    for t in tokens:

        # get word embedding
        emb = fast[t]

        # predict VAD - valence arousal dominance
        lemTokens.append(model.predict(np.array(emb).reshape(1, -1))[0])

    sentVAD = np.asarray(lemTokens)
    gobalVAD = np.sum(sentVAD, axis=0) / len(sentVAD)

    return gobalVAD.tolist()


def predict_by_lema(data, col, model, fast):
    '''
    :param data: dataframe
    :param col: string with column name
    :param model: keras machine learning model to do prediction
    :param fast: fasttext oject
    :return: Series with column treated
    '''

    data[col] = data[col].apply(lambda x: lemma2prediction(x, model, fast))
    return data[col]



def token_lematize(tokens):
    lemmatizer = WordNetLemmatizer()
    lemTokens = [lemmatizer.lemmatize(t) for t in tokens]
    return lemTokens


def lematize(dt, cl):
    dt[cl] = dt[cl].apply(lambda x: token_lematize(x))
    return dt[cl]


def load_fastext():
    '''
    :return: fasttext: fasttext english embedding
    '''

    return fasttext.load_model('/Users/anacosta/Desktop/KaggleComp_DonorsChoose/' + '/embeddings/wiki.en.bin')


def custom_loss(x,y):
    if x == -1:
        return 0
    else:
        # return K.switch(x[2] < 0, 0, losses.mse(x, y))
        return losses.mse(x, y)


# ---------- Cleaning datasets

def encodeColumn(data, col_name):
    # creating instance of labelencoder
    labelencoder = LabelEncoder()

    # Assigning numerical values and storing in another column
    data[col_name] = labelencoder.fit_transform(data[col_name])
    return data, labelencoder


def encodeDataset(projects, encoders_test=None):

    # Replace NaN values with students_reached column mode
    projects['students_reached'].fillna(projects['students_reached'].mode()[0], inplace=True)

    if encoders_test == None:
        projects, encoder_school_state = encodeColumn(projects, 'school_state')
        projects, encoder_teacher_acctid = encodeColumn(projects, 'teacher_acctid')
        projects, encoder_schoolid = encodeColumn(projects, 'schoolid')
        projects, encoder_school_ncesid = encodeColumn(projects, 'school_ncesid')
        projects, encoder_school_city = encodeColumn(projects, 'school_city')
        projects, encoder_school_metro = encodeColumn(projects, 'school_metro')
        projects, encoder_teacher_prefix = encodeColumn(projects, 'teacher_prefix')

        projects, encoder_secondary_focus_subject = encodeColumn(projects, 'secondary_focus_subject')
        projects, encoder_secondary_focus_area = encodeColumn(projects, 'secondary_focus_area')

        projects, encoder_primary_focus_subject = encodeColumn(projects, 'primary_focus_subject')
        projects, encoder_primary_focus_area = encodeColumn(projects, 'primary_focus_area')

        projects, encoder_resource_type = encodeColumn(projects, 'resource_type')
        projects, encoder_poverty_level = encodeColumn(projects, 'poverty_level')
        projects, encoder_grade_level = encodeColumn(projects, 'grade_level')

        projects, encoder_vendor_name = encodeColumn(projects, 'vendor_name')

        return projects, [encoder_school_state, encoder_teacher_acctid, encoder_schoolid, encoder_school_ncesid,
                          encoder_school_city, encoder_school_metro, encoder_teacher_prefix,
                          encoder_secondary_focus_subject, encoder_secondary_focus_area,
                          encoder_primary_focus_subject, encoder_primary_focus_area,
                          encoder_resource_type, encoder_poverty_level, encoder_grade_level,
                          encoder_vendor_name]


    else:

        projects['school_state'] = encoders_test[0].transform(projects['school_state'])
        projects['teacher_acctid'] = encoders_test[1].transform(projects['teacher_acctid'])
        projects['schoolid'] = encoders_test[2].transform(projects['schoolid'])
        projects['school_ncesid'] = encoders_test[3].transform(projects['school_ncesid'])
        projects['school_city'] = encoders_test[4].transform(projects['school_city'])
        projects['school_metro'] = encoders_test[5].transform(projects['school_metro'])
        projects['teacher_prefix'] = encoders_test[6].transform(projects['teacher_prefix'])
        projects['secondary_focus_subject'] = encoders_test[7].transform(projects['secondary_focus_subject'])
        projects['secondary_focus_area'] = encoders_test[8].transform(projects['secondary_focus_area'])
        projects['primary_focus_subject'] = encoders_test[9].transform(projects['primary_focus_subject'])
        projects['primary_focus_area'] = encoders_test[10].transform(projects['primary_focus_area'])
        projects['resource_type'] = encoders_test[11].transform(projects['resource_type'])
        projects['poverty_level'] = encoders_test[12].transform(projects['poverty_level'])
        projects['grade_level'] = encoders_test[13].transform(projects['grade_level'])

        projects['vendor_name'] = encoders_test[14].transform(projects['vendor_name'])

        return projects









