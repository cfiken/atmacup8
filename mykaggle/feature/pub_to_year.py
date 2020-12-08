from typing import Optional, Dict
import pandas as pd

from mykaggle.feature.base import Feature
from mykaggle.transform.groupby import BasicGroupByTransform


class PubToYear(Feature):
    '''
    Publisher の min/max/diff の year
    '''

    def __init__(self, train: bool = True) -> None:
        super().__init__(name='pub_to_year', train=train)

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

        agg_transform = BasicGroupByTransform(keys=['Publisher'], targets=['Year_of_Release'], aggs=['min'])
        df_agg = agg_transform(df_whole)
        df_main = pd.merge(df_main, df_agg, how='left', on='Publisher')
        min_column = df_agg.columns[-1]
        df_main['publisher_year_from_first'] = df_main.loc[:, 'Year_of_Release'] - df_main.loc[:, min_column]

        return df_main.loc[:, [min_column, 'publisher_year_from_first']]
