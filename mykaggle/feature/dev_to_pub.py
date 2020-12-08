from typing import Optional, Dict
import pandas as pd
from mykaggle.feature.base import Feature


class DevToPub(Feature):
    '''
    Developer から見た Publisher の重なり
    '''

    def __init__(self, train: bool = True) -> None:
        super().__init__(name='dev_to_pub', train=train)

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

        dev_to_pub = df_whole.groupby('Developer')['Publisher'].apply(list).reset_index()
        dev_to_pub['num_dev_to_publisher'] = dev_to_pub['Publisher'].apply(lambda x: len(x))
        dev_to_u_pub = df_whole.groupby('Developer')['Publisher'].apply(set).reset_index()
        dev_to_u_pub['num_dev_to_unique_publisher'] = dev_to_u_pub['Publisher'].apply(lambda x: len(list(x)))

        df_main = pd.merge(df_main, dev_to_pub, how='left', on='Developer')
        df_main = pd.merge(df_main, dev_to_u_pub, how='left', on='Developer')
        return df_main.loc[:, ['num_dev_to_publisher', 'num_dev_to_unique_publisher']]
