from typing import Optional, Dict
import pandas as pd
import numpy as np
from mykaggle.feature.base import Feature

COLUMNS = [
    'Critic_Score', 'User_Score', 'Critic_Count', 'User_Count'
]


class Score(Feature):
    '''
    Critic_Score, User_Score, Critic_Count, User_Count
    '''

    def __init__(self, train: bool = True) -> None:
        '''
        :params name: 特徴の名前 e.g.) gender, age
        :params train: train 用の特徴であれば True, test 用であれば False
        :params category: 特徴をまとめる dir を作る場合指定する e.g.) 特定コンペの名前など
        '''
        super().__init__(name='score', train=train)

    def create(
        self,
        base: pd.DataFrame,
        others: Optional[Dict[str, pd.DataFrame]] = None,
        *args, **kwargs
    ) -> pd.DataFrame:
        df_main = others['main'].copy()
        df_main.loc[df_main.loc[:, 'User_Score'] == 'tbd', 'User_Score'] = np.nan
        # normalize
        df_main.loc[:, 'User_Score'] = df_main.loc[:, 'User_Score'].astype(np.float32) / 10.0
        df_main.loc[:, 'Critic_Score'] = df_main.loc[:, 'Critic_Score'] / 100.0
        # df_main.loc[:, 'User_Score'] = df_main.loc[:, 'User_Score'].fillna(-1)
        # df_main.loc[:, 'Critic_Score'] = df_main.loc[:, 'Critic_Score'].fillna(-1)
        return df_main.loc[:, COLUMNS]
