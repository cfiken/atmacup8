from typing import Optional, Dict
import pandas as pd
from mykaggle.feature.base import Feature
from mykaggle.lib.pandas_util import change_column_name


class Index(Feature):
    '''
    配られた時点の index
    '''

    def __init__(self, train: bool = True) -> None:
        super().__init__(name='index', train=train)

    def create(
        self,
        base: pd.DataFrame,
        others: Optional[Dict[str, pd.DataFrame]] = None,
        *args, **kwargs
    ) -> pd.DataFrame:
        df_main = others['main'].copy()
        df_main = df_main.reset_index()
        df_main = change_column_name(df_main, 'index', 'original_index')
        return df_main.loc[:, ['original_index']]
