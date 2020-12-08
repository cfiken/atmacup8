from typing import Optional, Dict, List
import pandas as pd
import numpy as np
from collections import Counter
from sklearn.preprocessing import LabelEncoder

from mykaggle.feature.base import Feature


class PubToGenre(Feature):
    '''
    Publisher から見た Unique な Genre の数
    '''

    def __init__(self, train: bool = True) -> None:
        super().__init__(name='pub_to_genre', train=train)

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

        pub_to_genre = df_whole.groupby('Publisher')['Genre'].apply(set).reset_index()
        pub_to_genre['num_publisher_to_unique_genre'] = pub_to_genre['Genre'].apply(lambda x: len(list(x)))

        def genre_counter(x: List) -> str:
            return Counter(x).most_common()[0][0]

        df_main.loc[:, 'Genre'] = df_main.loc[:, 'Genre'].fillna("NaN")
        df_whole.loc[:, 'Genre'] = df_whole.loc[:, 'Genre'].fillna("NaN")
        pub_to_genre['publisher_big_genre'] = pub_to_genre.loc[:, 'Genre'].apply(genre_counter)
        pub_to_genre['publisher_big_genre'] = pub_to_genre.loc[:, 'publisher_big_genre'].fillna("NaN")
        pub_to_genre['le_publisher_big_genre'] = self._le_fit_transform(
            df_whole.loc[:, 'Genre'], pub_to_genre.loc[:, 'publisher_big_genre']
        )
        df_main['le_Genre'] = self._le_fit_transform(df_whole.loc[:, 'Genre'], df_main.loc[:, 'Genre'])
        df_main = pd.merge(df_main, pub_to_genre, how='left', on='Publisher')
        df_main['is_publisher_big_genre'] = df_main.loc[:, 'le_Genre'] == df_main.loc[:, 'le_publisher_big_genre']

        return df_main.loc[:, ['num_publisher_to_unique_genre', 'is_publisher_big_genre']]

    def _le_fit_transform(self, whole: pd.Series, x: pd.Series) -> np.ndarray:
        le = LabelEncoder()
        le.fit(whole)
        return le.transform(x)
