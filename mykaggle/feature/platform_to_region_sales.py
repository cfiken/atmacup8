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
    'Year_of_Release',
    # 'Genre',
    # 'Rating'
]


class PlatformToRegionSales(Feature):
    '''
    Platform ごとの Region 別 Sales の平均 (target encoding)
    '''

    def __init__(self, train: bool = True, n_components: int = 2) -> None:
        super().__init__(name='platform_to_region_sales', train=train)
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

        added_columns = []
        for c in COLUMNS:
            transform = BasicGroupByTransform(keys=[c], targets=SALES_COLUMNS, aggs=['mean'])
            platform_sales = transform(df_train)
            df_main = pd.merge(df_main, platform_sales, how='left', on=c)
            added_columns.extend(platform_sales.columns[1:])
        return df_main.loc[:, added_columns]
