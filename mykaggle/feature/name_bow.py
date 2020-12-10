from typing import Optional, Dict, Tuple
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer

from mykaggle.feature.base import Feature


class NameBOW(Feature):
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
        super().__init__(name='name_bow', train=train)
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

        def preprocess(x: str) -> str:
            x = x.replace(',', '')
            x = x.replace(':', ' ')
            x = x.replace('!', ' ')
            x = x.replace('\'s', ' ')
            x = x.replace('  ', ' ')
            return x.lower()

        df_whole.loc[:, 'processed_name'] = df_whole.loc[:, 'Name'].astype(str).apply(preprocess)
        df_main.loc[:, 'processed_name'] = df_main.loc[:, 'Name'].astype(str).apply(preprocess)
        unique_name = pd.DataFrame(df_whole['processed_name'].unique(), columns=['processed_name'])

        bow1 = self._get_bow(unique_name.loc[:, 'processed_name'].values,
                             (1, 1), self.word_count_th1, self.threshold_upper1)
        bow2 = self._get_bow(unique_name.loc[:, 'processed_name'].values,
                             (2, 4), self.word_count_th2, self.threshold_upper2)
        bow = np.concatenate([bow1, bow2], axis=1)
        df_pca = self._get_df_pca(bow, self.n_components, self.word_count_th1, self.word_count_th2)
        unique_name = pd.concat([unique_name, df_pca], axis=1)

        df_main = pd.merge(df_main, unique_name, how='left', on='processed_name')

        return df_main.loc[:, df_pca.columns]

    def _get_bow(
        self,
        texts: np.ndarray,
        ngram_range: Tuple[int, int],
        word_count_th: int,
        th_upper: int = 1000000
    ) -> np.ndarray:
        vectorizer = CountVectorizer(ngram_range=ngram_range)
        bow = vectorizer.fit_transform(texts)
        word_count = bow.toarray().sum(axis=0)
        bow = bow.toarray()[:, (word_count > word_count_th) & (word_count < th_upper)]
        return bow

    def _get_df_pca(self, bow: np.ndarray, n_components: int, th1: int, th2: int) -> pd.DataFrame:
        pca = PCA(n_components)
        bow_pca = pca.fit_transform(bow)
        pca_columns = [f'pca_{n}_name_bow_gt_count_{th1}_{th2}' for n in range(n_components)]
        return pd.DataFrame(bow_pca, columns=pca_columns)
