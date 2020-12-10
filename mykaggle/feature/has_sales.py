from typing import Optional, Dict
import pandas as pd
from mykaggle.feature.base import Feature

COLUMNS = [
    'has_jp_sales',
    'has_na_sales',
    'has_eu_sales',
    'has_other_sales',
]


class HasSales(Feature):
    '''
    元データの has_xx_sales, データが追加された country_prob のデータが必要
    '''

    def __init__(self, train: bool = True) -> None:
        super().__init__(name='has_sales', train=train)

    def create(
        self,
        base: pd.DataFrame,
        others: Optional[Dict[str, pd.DataFrame]] = None,
        *args, **kwargs
    ) -> pd.DataFrame:
        df_main = others['main'].copy()
        return df_main.loc[:, COLUMNS]
