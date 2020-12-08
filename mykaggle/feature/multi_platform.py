from typing import Optional, Dict
import pandas as pd
from mykaggle.feature.base import Feature
from mykaggle.lib.pandas_util import change_column_name


class MultiPlatform(Feature):
    '''
    Name の Platform 被りを数える
    '''

    def __init__(self, train: bool = True) -> None:
        super().__init__(name='multi_platform', train=train)

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

        multi_platform = (df_whole.loc[:, ['Name']].value_counts()).reset_index()
        multi_platform = change_column_name(multi_platform, 0, 'multi_pf_count')
        df_main = pd.merge(df_main, multi_platform, how='left', on='Name')
        return df_main.loc[:, ['multi_pf_count']]
