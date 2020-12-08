from typing import Optional, Dict
import numpy as np
import pandas as pd
from mykaggle.feature.base import Feature
from mykaggle.transform.groupby import BasicGroupByTransform

COLUMNS = ['Year_of_Release']


class YearRank(Feature):
    '''
    Year ごとの rank. どうやら順序が発売順になってそうな雰囲気なので
    '''

    def __init__(self, train: bool = True) -> None:
        super().__init__(name='year_rank', train=train)

    def create(
        self,
        base: pd.DataFrame,
        others: Optional[Dict[str, pd.DataFrame]] = None,
        *args, **kwargs
    ) -> pd.DataFrame:
        df_main = others['main'].copy()

        count_transformer = BasicGroupByTransform(keys=['Year_of_Release'], targets=['id'], aggs=['count'])
        count_agg = count_transformer(df_main)
        df_main = pd.merge(df_main, count_agg, how='left', on='Year_of_Release')

        year_rank = df_main.groupby('Year_of_Release')['id'].rank()
        df_main['year_rank'] = year_rank

        df_main.loc[df_main.loc[:, count_agg.columns[-1]].isna(), 'year_rank'] = np.nan
        df_main['year_rank_rate'] = df_main.loc[:, 'year_rank'] / df_main.loc[:, count_agg.columns[-1]]
        return df_main.loc[:, ['year_rank_rate']]
