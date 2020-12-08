from typing import Optional, Dict
import pandas as pd
from mykaggle.feature.base import Feature
from mykaggle.transform.groupby import BasicGroupByTransform

COLUMNS = [
    'Platform',
    'Year_of_Release',
]


class TEYearPlatform(Feature):
    '''
    Name 以外の全カテゴリのカウントエンコーディング
    '''

    def __init__(self, train: bool = True) -> None:
        super().__init__(name='te_year_platform', train=train)

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

        mean_transformer = BasicGroupByTransform(COLUMNS, ['target'], ['mean', 'median', 'sum', 'std'])
        te = mean_transformer(df_train)
        df_main = pd.merge(df_main, te, how='left', on=COLUMNS)
        return df_main.loc[:, te.columns[-4:]]
