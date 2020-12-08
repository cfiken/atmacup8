from typing import Optional, Dict
import pandas as pd
from mykaggle.feature.base import Feature

COLUMNS = ['Year_of_Release']


class Original(Feature):
    '''
    元データママで使えるやつ
    '''

    def __init__(self, train: bool = True) -> None:
        super().__init__(name='original', train=train)

    def create(
        self,
        base: pd.DataFrame,
        others: Optional[Dict[str, pd.DataFrame]] = None,
        *args, **kwargs
    ) -> pd.DataFrame:
        df_main = others['main'].copy()
        return df_main.loc[:, COLUMNS]
