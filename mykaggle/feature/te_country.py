from typing import Optional, Dict
import pandas as pd
from mykaggle.feature.base import Feature
from mykaggle.transform.groupby import BasicGroupByTransform

COLUMNS = [
    'Platform',
    'Year_of_Release',
    'Genre',
]
SALES_COLUMNS = ['mod_NA_Sales', 'mod_EU_Sales', 'mod_JP_Sales', 'mod_Other_Sales']


class TECountry(Feature):
    '''
    各地域別の売上の target encoding
    '''

    def __init__(self, train: bool = True) -> None:
        super().__init__(name='te_country', train=train)

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
            mean_transformer = BasicGroupByTransform([c], SALES_COLUMNS, ['mean', 'median'])
            te = mean_transformer(df_train)
            df_main = pd.merge(df_main, te, how='left', on=c)
            added_columns.extend(te.columns[-len(SALES_COLUMNS):])
        return df_main.loc[:, added_columns]
