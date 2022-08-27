
from sklearn import preprocessing
import pandas as pd
from nltk.tokenize import  word_tokenize
import numpy as np
import time as tm

class TfIdf():

    def __init__(self, text_df, text_column_name):
        self.sentance_word_dict = {}
        self.text_df = text_df
        self.list_bag_of_word = self.split_text(self.text_df,text_column_name)
        self.set_bag_of_word_frequencies()
        self.word_frequencies_df = self.create_initial_df(text_column_name)
        self.tfIdf_df = self.create_tfIdf_df(self.word_frequencies_df)
        self.tfIdf_df_norm = self.normalize_numeric_df(self.tfIdf_df)

    def count_unique_word_in_sentance(self, sentance):
        """
        Counting the unique word in a sentance transform it into dictionary
        :param sentance:
        :return:
        """
        sentance_word_dict_temp = self.sentance_word_dict.copy()
        temp = pd.value_counts(np.array(word_tokenize(sentance))).to_dict()
        sentance_word_dict_temp.update(temp)

        return sentance_word_dict_temp

    @staticmethod
    def isnumber(x):
        try:
            if float(x) or  int(x):
                return True
        except:
            return False

    @staticmethod
    def remove_non_numeric_word(df):
        '''
        Transform non numeric values into non (should be only numeric) and validation for non NaN values
        '''
        df = df[df.applymap(TfIdf.isnumber)]
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0, downcast='infer')
        return df

    def split_text(self,df, text_column_name):
        """
        From corpus, transform the df of sentences into a bag of words
        :param df: Corpus
        :param text_column_name: Column name
        :return: bad of words in a list
        """

        sentance = ' '.join(df[text_column_name].tolist())
        set_words_in_sentance = word_tokenize(sentance)
        list_bag_of_word = set(set_words_in_sentance)

        return list_bag_of_word

    def create_initial_df(self, text_column_name):
        '''
        Create a frequency df which is the base for calculating the tfidf algorithm.
        First calculate list of frequency per sentance.
        Second convert it into data frame
        '''

        list_word_frequency_in_a_sentence = list(map(lambda x: self.count_unique_word_in_sentance(x),self.text_df[text_column_name]))
        df = pd.DataFrame.from_records(list_word_frequency_in_a_sentence)
        return df

    def set_bag_of_word_frequencies(self):
        if self.list_bag_of_word == 0:
            raise ValueError("Bage of word list is empty")
        else:
            self.sentance_word_dict = dict.fromkeys(self.list_bag_of_word,0)
        return

    def create_tfIdf_df(self,df):
        """
        Create df_Idf DataFrame DataFrame. dividing each value by the sum of the entire row, each row signify a document
        In the second step, it calculate the values of entire df. The result would be the relative ratio of a word in the df.
        A simple multiplication to receive the df_Idf DataFrame
        :param df:
        :return: tfIdf data frame
        """

        tf_df = df.divide(df.sum(axis=1),axis='index')
        count_word_in_all_documents = np.log(df.shape[1]/df.sum(axis=0))
        df_tfIdf = tf_df.mul(count_word_in_all_documents, axis=1)

        return  df_tfIdf

    def normalize_numeric_df(self,df_tfIdf):
        '''Normlizing the df less variances'''
        x = df_tfIdf.values #returns a numpy array
        min_max_scalar = preprocessing.MinMaxScaler()
        x_scaled = min_max_scalar.fit_transform(x)
        return  pd.DataFrame(x_scaled)

if __name__ == '__main__':
    train_filename = "train_file.xlsx"
    test_filename = "test_file.xlsx"
    df_train = pd.read_excel(train_filename,
                             engine='openpyxl')

    start = tm.time()
    obj = TfIdf(df_train, 'story')
    end = tm.time()

    print(('vectorizing took %d seconds to run')%(end-start))