from typing import Optional, Dict
import pandas as pd

from mykaggle.feature.base import Feature
from mykaggle.transform.groupby import BasicGroupByTransform

SALES_COLUMNS = [
    'JP_Sales',
    'NA_Sales',
    'EU_Sales',
    'Other_Sales',
]

COLUMNS = [
    'Platform',
    # 'Year_of_Release',
    # 'Genre',
    # 'Rating'
]


class PlatformToRegionHasSales(Feature):
    '''
    Platform を Publisher 情報からエンコード
    '''

    def __init__(self, train: bool = True, n_components: int = 2) -> None:
        super().__init__(name='platform_to_region_has_sales', train=train)
        self.n_components = n_components

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

        has_sales_columns = [f'has_{c}' for c in SALES_COLUMNS]
        for c in SALES_COLUMNS:
            df_train[f'has_{c}'] = df_train.loc[:, c].fillna(0) > 0

        added_columns = []
        for c in COLUMNS:
            transform = BasicGroupByTransform(keys=[c], targets=has_sales_columns, aggs=['mean'])
            platform_sales_mean = transform(df_train)
            df_main = pd.merge(df_main, platform_sales_mean, how='left', on=c)
            added_columns.extend(platform_sales_mean.columns[1:])
        return df_main.loc[:, added_columns]
