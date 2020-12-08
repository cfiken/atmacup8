from typing import Optional, Dict
import pandas as pd
from mykaggle.feature.base import Feature
from mykaggle.transform.groupby import BasicGroupByTransform

COLUMNS = [
    'Publisher'
]


class CEPublisher(Feature):
    '''
    Publisher のカウントエンコーディング
    '''

    def __init__(self, train: bool = True) -> None:
        super().__init__(name='ce_publisher', train=train)

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

        added_columns = []
        for c in COLUMNS:
            count_transformer = BasicGroupByTransform([c], ['id'], ['count'])
            count_agg = count_transformer(df_whole)
            df_main = pd.merge(df_main, count_agg, how='left', on=c)
            added_columns.append(count_agg.columns[-1])
        return df_main.loc[:, added_columns]
