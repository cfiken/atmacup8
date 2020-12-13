from typing import Optional, Dict
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from mykaggle.feature.base import Feature

COLUMNS = [
    'Publisher',
]


class LEPublisher(Feature):
    '''
    Name 以外の全カテゴリのラベルエンコーディング
    '''

    def __init__(self, train: bool = True) -> None:
        '''
        :params name: 特徴の名前 e.g.) gender, age
        :params train: train 用の特徴であれば True, test 用であれば False
        :params category: 特徴をまとめる dir を作る場合指定する e.g.) 特定コンペの名前など
        '''
        super().__init__(name='le_publisher', train=train)

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
        for c in COLUMNS:
            df_whole.loc[:, c] = df_whole.loc[:, c].fillna("NaN")
            df_main.loc[:, c] = df_main.loc[:, c].fillna("NaN")
            df_main['le_' + c] = self.fit_transform_le(df_whole.loc[:, c], df_main.loc[:, c])
        columns = ['le_' + c for c in COLUMNS]
        return df_main.loc[:, columns]

    def fit_transform_le(self, whole: pd.Series, x: pd.Series):
        le = LabelEncoder()
        le.fit(whole)
        return le.transform(x)
