import pandas as pd
from sklearn.base import TransformerMixin
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from wordcloud import WordCloud
import numpy as np


class Custom_Transformer(BaseEstimator, TransformerMixin):
    def __init__(self,speaker_1,speaker_2,target_1,target_2):
        self.speaker_1 = speaker_1
        self.speaker_2 = speaker_2
        self.target_1 = target_1
        self.target_2 = target_2
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        df_speaker1 = X[X['speaker']==self.speaker_1]
        df_speaker2 = X[X['speaker']==self.speaker_2]
        df_target_1 = pd.DataFrame(len(df_speaker1)*[self.target_1],index=df_speaker1.index)
        df_target_2 = pd.DataFrame(len(df_speaker2)*[self.target_2],index=df_speaker2.index)
        df_speaker1 = df_speaker1.assign(target=df_target_1)
        df_speaker2 = df_speaker2.assign(target=df_target_2)
        df_concat = pd.concat([df_speaker1,df_speaker2],axis=0)
        return df_concat




def get_polar_words(classifier):
    coeff_pos = classifier['Multinomial NB'].feature_log_prob_[1]
    coeff_neg = classifier['Multinomial NB'].feature_log_prob_[0]
    words = classifier['Tfidf vectorizer'].get_feature_names_out()
    polarity = coeff_pos-coeff_neg
    polarity_sorted = np.sort(polarity)
    index_sort = np.argsort(polarity)
    words_sorted = words[index_sort]
    words_positive = words_sorted[-100:]
    words_negative = words_sorted[:100]
    wc1 = WordCloud(width = 1000, height = 500).generate(" ".join(words_positive))
    wc2 = WordCloud(width = 1000, height = 500).generate(" ".join(words_negative))
    return wc1,wc2
    
