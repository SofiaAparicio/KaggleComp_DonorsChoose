import string
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


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


def token_lematize(tokens):
    lemmatizer = WordNetLemmatizer()
    lemTokens = [lemmatizer.lemmatize(t) for t in tokens]
    return lemTokens


def lematize(dt, cl):
    dt[cl] = dt[cl].apply(lambda x: token_lematize(x))
    return dt[cl]














