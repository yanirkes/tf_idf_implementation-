from sklearn import preprocessing
import pandas as pd
from nltk.tokenize import  word_tokenize
import numpy as np
import openpyxl

class TfIdf():

    def __init__(self, text_df, text_column_name):
        self.text_df = text_df
        self.list_bag_of_word, self.list_of_sentanceWord_dict = self.split_text(self.text_df,text_column_name)
        self.word_frequencies_df = self.create_initial_df(self.list_of_sentanceWord_dict, self.list_bag_of_word)
        self.tfIdf_df = self.create_tfIdf_df(self.word_frequencies_df)
        self.tfIdf_df_norm = self.normalize_numeric_df(self.tfIdf_df)

    def vectorize(self, data, tfidf_vect_fit):
        X_tfidf = tfidf_vect_fit.transform(data)
        words = tfidf_vect_fit.get_feature_names()
        X_tfidf_df = pd.DataFrame(X_tfidf.toarray())
        X_tfidf_df.columns = words
        return (X_tfidf_df)

    @staticmethod
    def count_unique_word_in_sentance(bag_of_word, sentance):
      sentance_word_dict = dict.fromkeys(bag_of_word,0)
      for word in sentance:
        sentance_word_dict[word]+=1
      return sentance_word_dict

    @staticmethod
    def isnumber(x):
        try:
            if float(x) or  int(x):
                return True
        except:
            return False

    @staticmethod
    def remove_non_numeric_word(df):
        '''Transform non numeric values into non (should be only numeric) and validation for non NaN values'''
        df = df[df.applymap(TfIdf.isnumber)]
        df = df.apply(pd.to_numeric, errors='coerce').fillna(0, downcast='infer')
        return df


    # model func
    def split_text(self,df, text_column_name):
        '''Main taget its to create 2 objects - a bag of words of the entire data set,
            and to ceate a list of dic such that each dict will count the word appearances in a single
            text (a row)
        '''

        list_bag_of_word = {}
        list_of_sentanceWord_dict = []

        for sentance in df[text_column_name]:
            set_words_in_sentance = word_tokenize(sentance)
            list_of_sentanceWord_dict.append(set_words_in_sentance)
            list_bag_of_word = set(list_bag_of_word)
            list_bag_of_word = list_bag_of_word.union(set_words_in_sentance)
        return list_bag_of_word, list_of_sentanceWord_dict

    def create_initial_df(self,list_of_sentanceWord_dict, list_bag_of_word):
        '''Create a df that will provide the ground base for calculating the tfidf algorithm'''
        list_word_frequency_in_a_sentence = []
        for sentance in list_of_sentanceWord_dict:
            temp = self.count_unique_word_in_sentance(list_bag_of_word, sentance)
            list_word_frequency_in_a_sentence.append(temp)
        del temp

        df = pd.DataFrame.from_records(list_word_frequency_in_a_sentence)
        df = self.remove_non_numeric_word(df)
        return df

    def create_tfIdf_df(self,df):
        ''' Create df_Idf DataFrame DataFrame. dividing each value by the sum of the entire row, each row signify a document
            In the second step, it calculate the values of entire df. The result would be the relative ratio of a word in the df.
            A simple multiplication to receive the df_Idf DataFrame'''
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

    obj = TfIdf(df_train, 'story')
