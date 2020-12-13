from typing import Optional, Dict
import pandas as pd
import numpy as np
from mykaggle.feature.base import Feature
from mykaggle.transform.groupby import BasicGroupByTransform

COLUMNS = [
    'Critic_Score', 'User_Score', 'Critic_Count', 'User_Count'
]


class ScoreNaN(Feature):
    '''
    Critic_Score, User_Score, Critic_Count, User_Count
    '''

    def __init__(self, train: bool = True) -> None:
        '''
        :params name: 特徴の名前 e.g.) gender, age
        :params train: train 用の特徴であれば True, test 用であれば False
        :params category: 特徴をまとめる dir を作る場合指定する e.g.) 特定コンペの名前など
        '''
        super().__init__(name='score_nan', train=train)

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

        df_main.loc[df_main.loc[:, 'User_Score'] == 'tbd', 'User_Score'] = np.nan
        df_whole.loc[df_whole.loc[:, 'User_Score'] == 'tbd', 'User_Score'] = np.nan

        # normalize
        df_main.loc[:, 'User_Score'] = df_main.loc[:, 'User_Score'].astype(np.float32) / 10.0
        df_main.loc[:, 'Critic_Score'] = df_main.loc[:, 'Critic_Score'] / 100.0
        df_whole.loc[:, 'User_Score'] = df_whole.loc[:, 'User_Score'].astype(np.float32) / 10.0
        df_whole.loc[:, 'Critic_Score'] = df_whole.loc[:, 'Critic_Score'] / 100.0

        transform = BasicGroupByTransform(keys=['Platform', 'Year_of_Release'], targets=COLUMNS, aggs=['mean'])
        platform_scores = transform(df_whole)
        df_main = pd.merge(df_main, platform_scores, how='left', on=['Platform', 'Year_of_Release'])

        for c in COLUMNS:
            c_na = df_main[c].isna()
            df_main.loc[c_na, c] = df_main.loc[c_na, f'mean_{c}_groupby_Platform_Year_of_Release']
        return df_main.loc[:, COLUMNS]
