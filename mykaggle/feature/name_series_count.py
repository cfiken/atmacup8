from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
import texthero
from texthero import preprocessing

from mykaggle.lib.pandas_util import change_column_name
from mykaggle.feature.base import Feature


class NameSeriesCount(Feature):
    '''
    名前の BoW, 1単語と複数単語で分けた。1単語でたくさん出てるやつも入れる。
    '''

    def __init__(
        self,
        train: bool = True,
        n_components: int = 4,
        word_count_th1: int = 5,
        word_count_th2: int = 3,
        threshold_upper1: int = 100000,
        threshold_upper2: int = 100000,
    ) -> None:
        super().__init__(name='name_series_count', train=train)
        self.n_components = n_components
        self.word_count_th1 = word_count_th1
        self.word_count_th2 = word_count_th2
        self.threshold_upper1 = threshold_upper1
        self.threshold_upper2 = threshold_upper2

    def create(
        self,
        base: pd.DataFrame,
        others: Optional[Dict[str, pd.DataFrame]] = None,
        *args, **kwargs
    ) -> pd.DataFrame:
        df_main = others['main'].copy()
        df_another = others['another'].copy()
        if self.train:
            df_whole = pd.concat([df_main, df_another])
        else:
            df_whole = pd.concat([df_another, df_main])

        df_whole.loc[:, 'clean_name'] = self._clean(df_whole.loc[:, 'Name'])
        df_main.loc[:, 'clean_name'] = self._clean(df_main.loc[:, 'Name'])

        series_count1 = self._get_top_text_ngrams(df_whole.loc[:, 'clean_name'], 100000, (1, 1), 50)
        series_count1 = dict(series_count1[::-1])
        series_count2 = self._get_top_text_ngrams(df_whole.loc[:, 'clean_name'], 100000, (2, 2), 20)
        series_count2 = dict(series_count2[::-1][:-2])
        series_count3 = self._get_top_text_ngrams(df_whole.loc[:, 'clean_name'], 100000, (3, 3), 10)
        series_count3 = dict(series_count3[::-1])

        df_main['num_word_series_1'] = 0
        for i in series_count1:
            if len(i) < 5:
                continue
            idx = df_main[df_main.loc[:, 'clean_name'].str.startswith(i)].index
            df_main.loc[idx, 'num_word_series_1'] = series_count1[i]
        df_main['num_word_series_2'] = 0
        for i in series_count2:
            idx = df_main[df_main.loc[:, 'clean_name'].str.startswith(i)].index
            df_main.loc[idx, 'num_word_series_2'] = series_count2[i]
        df_main['num_word_series_3'] = 0
        for i in series_count3:
            idx = df_main[df_main.loc[:, 'clean_name'].str.contains(i)].index
            df_main.loc[idx, 'num_word_series_3'] = series_count3[i]

        name_platform = df_whole.groupby('Name')['Platform'].nunique()
        name_platform = change_column_name(name_platform, 'Platform', 'nunique_platform')
        df_main = pd.merge(df_main, name_platform, how='left', on='Name')
        df_main['num_word_series_1'] = df_main.loc[:, 'num_word_series_1'] / df_main.loc[:, 'nunique_platform']
        df_main['num_word_series_2'] = df_main.loc[:, 'num_word_series_2'] / df_main.loc[:, 'nunique_platform']
        df_main['num_word_series_3'] = df_main.loc[:, 'num_word_series_3'] / df_main.loc[:, 'nunique_platform']

        df_main.loc[df_main['num_word_series_1'] == 0, 'num_word_series_1'] = 1
        df_main.loc[df_main['num_word_series_2'] == 0, 'num_word_series_2'] = 1
        df_main.loc[df_main['num_word_series_3'] == 0, 'num_word_series_3'] = 1

        return df_main.loc[:, ['num_word_series_1', 'num_word_series_2', 'num_word_series_3']]

    def _get_top_text_ngrams(self, corpus: np.ndarray, n: int, g: Tuple[int, int], s: int):
        vec = CountVectorizer(ngram_range=g).fit(corpus)
        bag_of_words = vec.transform(corpus)
        sum_words = bag_of_words.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items() if sum_words[0, idx] > s]
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        return words_freq[:n]

    def _clean(self, series: pd.Series) -> pd.Series:
        custom_pipeline = [
            preprocessing.fillna,
            preprocessing.lowercase,
            preprocessing.remove_digits,
            preprocessing.remove_punctuation,
            preprocessing.remove_diacritics,
            preprocessing.remove_whitespace
        ]
        return texthero.clean(series, pipeline=custom_pipeline)
