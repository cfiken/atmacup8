from typing import Optional, Dict
import pandas as pd
from mykaggle.feature.base import Feature
from mykaggle.transform.groupby import BasicGroupByTransform

COLUMNS = [
    'Developer',
    'Genre',
    'Platform',
    'Year_of_Release'
]


class PubToCategory(Feature):
    '''
    Publisher から見た各カテゴリのユニーク数
    '''

    def __init__(self, train: bool = True) -> None:
        super().__init__(name='pub_to_category', train=train)

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

        agg_columns = []
        for c in COLUMNS:
            transform = BasicGroupByTransform(keys=['Publisher'], targets=[c], aggs=['nunique'])
            pub_to_c = transform(df_whole)
            agg_columns.append(pub_to_c.columns[-1])
            df_main = pd.merge(df_main, pub_to_c, how='left', on='Publisher')

        return df_main.loc[:, agg_columns]
