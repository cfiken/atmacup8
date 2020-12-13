from typing import Optional, Dict
import pandas as pd

from mykaggle.lib.pandas_util import change_column_name
from mykaggle.feature.base import Feature


class PlatformKing(Feature):
    '''
    Platform でスコアの高い Publisher フラグ
    '''

    def __init__(self, train: bool = True) -> None:
        super().__init__(name='platform_king', train=train)

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

        df_platform = self.get_platform_rank(df_whole)
        df_platform = df_platform[df_platform.loc[:, 'rank'] <= 3.0]
        df_platform = df_platform[df_platform.loc[:, 'platform_score'] >= df_platform.loc[:, 'max_score'] * 0.6]
        df_platform = df_platform[df_platform.loc[:, 'platform_score'] >= 10.0]
        df_platform['platform_king'] = 1
        df_platform = df_platform[['Platform', 'Publisher', 'platform_king']]

        df_main = pd.merge(df_main, df_platform, how='left', on=['Platform', 'Publisher'])
        df_main['platform_king'] = df_main['platform_king'].fillna(0)
        return df_main.loc[:, ['platform_king']]

    def get_platform_rank(self, df: pd.DataFrame) -> pd.DataFrame:
        df_platform = df.groupby(['Platform', 'Publisher'])['Name'].count().reset_index()
        df_platform = change_column_name(df_platform, 'Name', 'count')
        df_pub_platform_target_mean = df.groupby(['Platform', 'Publisher'])['Global_Sales'].mean().reset_index()
        df_pub_platform_target_mean = change_column_name(df_pub_platform_target_mean, 'Global_Sales', 'mean_target')
        df_platform = pd.merge(df_platform, df_pub_platform_target_mean, how='left', on=['Platform', 'Publisher'])
        df_platform.loc[:, 'platform_score'] = df_platform.loc[:, 'count'] * df_platform.loc[:, 'mean_target']
        df_pub_platform_max = df_platform.groupby('Platform')['platform_score'].max().reset_index()
        df_pub_platform_max = change_column_name(df_pub_platform_max, 'platform_score', 'max_score')
        df_platform = pd.merge(df_platform, df_pub_platform_max, how='left', on='Platform')
        df_platform['rank'] = df_platform.groupby(['Platform'])['platform_score'].rank(ascending=False)
        return df_platform
