from typing import Optional, Dict
import pandas as pd
from mykaggle.feature.base import Feature
from mykaggle.transform.groupby import BasicGroupByTransform

COLUMNS = [
    'Platform',
    'Year_of_Release',
    'Genre',
    'Rating'
]


class TE(Feature):
    '''
    Name 以外の全カテゴリのカウントエンコーディング
    '''

    def __init__(self, train: bool = True) -> None:
        super().__init__(name='te', train=train)

    def create(
        self,
        base: pd.DataFrame,
        others: Optional[Dict[str, pd.DataFrame]] = None,
        *args, **kwargs
    ) -> pd.DataFrame:
        df_main = others['main'].copy()
        if self.train:
            df_train = others['main'].copy()
        else:
            df_train = others['another'].copy()

        added_columns = []
        for c in COLUMNS:
            mean_transformer = BasicGroupByTransform([c], ['target'], ['mean'])
            te = mean_transformer(df_train)
            df_main = pd.merge(df_main, te, how='left', on=c)
            added_columns.append(te.columns[-1])
        return df_main.loc[:, added_columns]
