from typing import Optional, Dict
import pandas as pd
from mykaggle.feature.base import Feature
from mykaggle.feature.year_rank5 import YearRank5

COLUMNS = [
    'Platform',
    'Genre',
    'Developer',
    'Rating'
]


class PlatformTimeDiff(Feature):
    '''
    Name 以外の全カテゴリのラベルエンコーディング
    '''

    def __init__(self, train: bool = True) -> None:
        super().__init__(name='platform_time_diff', train=train)

    def create(
        self,
        base: pd.DataFrame,
        others: Optional[Dict[str, pd.DataFrame]] = None,
        *args, **kwargs
    ) -> pd.DataFrame:
        df_main = others['main'].copy()
        df_another = others['another'].copy()
        year_rank = YearRank5(train=self.train)
        df_main = year_rank(df_main, others=others, use_cache=False, save_cache=False)
        df_another = year_rank(df_another, others=others, use_cache=False, save_cache=False)
        if self.train:
            df_whole = pd.concat([df_main, df_another])
        else:
            df_whole = pd.concat([df_another, df_main])
        pf_to_yor = df_whole.groupby('Platform')['year_rank_plus'].agg(['min', 'max']).reset_index()
        pf_to_yor.columns = ['Platform', 'platform_year_rank_min', 'platform_year_rank_max']
        df_main = pd.merge(df_main, pf_to_yor, how='left', on='Platform')
        df_main['diff_platform_min'] = df_main.loc[:, 'year_rank_plus'] - df_main.loc[:, 'platform_year_rank_min']
        df_main['diff_platform_max'] = df_main.loc[:, 'platform_year_rank_max'] - df_main.loc[:, 'year_rank_plus']

        return df_main.loc[:, ['diff_platform_min']]
