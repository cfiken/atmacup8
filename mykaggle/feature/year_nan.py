from typing import Optional, Dict
import pandas as pd

from mykaggle.feature.base import Feature


class YearNaN(Feature):
    '''
    Year が NaN のところに Platform での最頻値を入れる
    '''

    def __init__(self, train: bool = True, n_components: int = 2) -> None:
        super().__init__(name='year_nan', train=train)
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

        platform_year_mode = df_whole.groupby('Platform')['Year_of_Release'].agg(lambda x: x.value_counts().index[0])
        platform_year_mode_map = dict(platform_year_mode)
        year_na = df_main['Year_of_Release'].isna()
        df_main.loc[year_na, 'Year_of_Release'] = df_main.loc[year_na,
                                                              'Platform'].apply(lambda x: platform_year_mode_map[x])

        return df_main.loc[:, ['Year_of_Release']]
