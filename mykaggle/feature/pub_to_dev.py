from typing import Optional, Dict
import pandas as pd
from mykaggle.feature.base import Feature


class PubToDev(Feature):
    '''
    Developer から見た Publisher の重なり
    '''

    def __init__(self, train: bool = True) -> None:
        super().__init__(name='pub_to_dev', train=train)

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

        pub_to_dev = df_whole.groupby('Publisher')['Developer'].apply(set).reset_index()
        pub_to_dev['num_publisher_to_unique_dev'] = pub_to_dev['Developer'].apply(lambda x: len(list(x)))

        df_main = pd.merge(df_main, pub_to_dev, how='left', on='Publisher')
        return df_main.loc[:, ['num_publisher_to_unique_dev']]
