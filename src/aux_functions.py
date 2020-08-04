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
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from geopy.geocoders import Nominatim
from collections import Counter
import category_encoders as ce
from sklearn.metrics import f1_score, balanced_accuracy_score, roc_curve




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

def encodeTarget(data, target, col_names):
    # creating an instance of target encoder
    target_encoder = ce.TargetEncoder(cols=col_names, min_samples_leaf=1, return_df=True,
                                      drop_invariant=False)

    target_encoder.fit(data, target)
    data = target_encoder.transform(data)

    return data, target_encoder


def encodeOrdinal(data, col_names):
    # creating instance of encoder
    ordinal_encoder = OrdinalEncoder()

    # Assigning numerical values and storing in another column
    ordinal_encoder.fit(data[col_names])
    data[col_names] = ordinal_encoder.transform(data[col_names])
    return data, ordinal_encoder



def encodeDataset(projects, target=None, encoders_test=None):

    # Replace NaN values with students_reached column mode
    projects['students_reached'].fillna(projects['students_reached'].mode()[0], inplace=True)


    if encoders_test == None:

        projects, ordinal_encoder = encodeOrdinal(projects, ['poverty_level', 'grade_level'])

        projects, target_encoder = encodeTarget(projects, target, ['school_state', 'teacher_acctid',
                                                           'schoolid', 'school_city',
                                                           'school_metro', 'teacher_prefix', 'secondary_focus_subject',
                                                           'secondary_focus_area', 'primary_focus_subject',
                                                           'primary_focus_area', 'resource_type', 'vendor_name'])

        # regarding the labels on the test set that are only seen once, the encoder will give them a NaN value
        # tranform the NaN values to 0
        projects = projects.replace(np.nan, 0, regex=True)

        return projects, [ordinal_encoder, target_encoder]


    else:

        projects[['poverty_level', 'grade_level']] = encoders_test[0].transform(projects[['poverty_level', 'grade_level']])
        projects = encoders_test[1].transform(projects)

        # regarding the unseen labels on the test set, the encoder will give them a NaN value
        # tranform the NaN values to 0
        projects = projects.replace(np.nan, 0, regex=True)

        return projects


def classificationReport(validation_y, validation_random_under_Predict):
    f1score_random_under = f1_score(validation_y['is_exciting'], validation_random_under_Predict, average='weighted')

    balaccu_random_under = balanced_accuracy_score(validation_y['is_exciting'], validation_random_under_Predict)

    # Evaluation metrics
    print('Scores are:')

    print('F1 score : {} '.format(f1score_random_under))
    print('Balanced Accuracy score : {} '.format(balaccu_random_under))





