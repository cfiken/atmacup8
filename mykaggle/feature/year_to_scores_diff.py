from typing import Optional, Dict
import pandas as pd
import numpy as np

from mykaggle.feature.base import Feature
from mykaggle.transform.groupby import BasicGroupByTransform

TARGET_COLUMNS = [
    'User_Score',
    'User_Count',
    'Critic_Score',
    'Critic_Count'
]


class YearToScoresDiff(Feature):
    '''
    Year についてUserやCriticsのスコア、数の情報
    '''

    def __init__(self, train: bool = True, n_components: int = 2) -> None:
        super().__init__(name='year_to_scores_diff', train=train)
        self.n_components = n_components

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

        df_whole.loc[df_whole.loc[:, 'User_Score'] == 'tbd', 'User_Score'] = np.nan
        df_main.loc[df_main.loc[:, 'User_Score'] == 'tbd', 'User_Score'] = np.nan
        df_whole.loc[:, 'User_Score'] = df_whole.loc[:, 'User_Score'].astype(np.float32) / 10.0
        df_main.loc[:, 'User_Score'] = df_main.loc[:, 'User_Score'].astype(np.float32) / 10.0
        df_whole.loc[:, 'Critic_Score'] = df_whole.loc[:, 'Critic_Score'] / 100.0
        df_main.loc[:, 'Critic_Score'] = df_main.loc[:, 'Critic_Score'] / 100.0
        transform = BasicGroupByTransform(keys=['Year_of_Release'], targets=TARGET_COLUMNS, aggs=['mean', 'max', 'min'])
        platform_scores = transform(df_whole)
        df_main = pd.merge(df_main, platform_scores, how='left', on='Year_of_Release')
        agg_columns = platform_scores.columns[1:]
        feature_columns = []
        for c in agg_columns:
            for tc in TARGET_COLUMNS:
                if tc in c:
                    column_name = f'diff_{tc}_{c}'
                    df_main[column_name] = df_main.loc[:, tc] - df_main.loc[:, c]
                    feature_columns.append(column_name)
        return df_main.loc[:, feature_columns]
