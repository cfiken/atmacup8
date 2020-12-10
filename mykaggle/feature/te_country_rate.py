from typing import Optional, Dict
import pandas as pd
from mykaggle.feature.base import Feature
from mykaggle.transform.groupby import BasicGroupByTransform

COLUMNS = [
    'Platform',
    # 'Year_of_Release',
    # 'Genre',
]
SALES_COLUMNS = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']


class TECountryRate(Feature):
    '''
    各地域別の売上の target encoding
    '''

    def __init__(self, train: bool = True) -> None:
        super().__init__(name='te_country_rate', train=train)

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
            transform = BasicGroupByTransform([c], SALES_COLUMNS, ['sum'])
            df_sum = transform(df_train)
            global_column = df_sum.columns[-1]
            country_columns = df_sum.columns[1:-1]
            for cc in country_columns:
                df_sum[f'{cc}_rate'] = df_sum.loc[:, cc] / df_sum.loc[:, global_column]
            df_main = pd.merge(df_main, df_sum, how='left', on=c)
            country_columns = [f'{cc}_rate' for cc in country_columns]
            added_columns.extend(country_columns)
        return df_main.loc[:, added_columns]
