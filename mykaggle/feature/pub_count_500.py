from typing import Optional, Dict
import pandas as pd
from mykaggle.feature.base import Feature
from mykaggle.transform.groupby import BasicGroupByTransform


class PubCount500(Feature):
    '''
    Publisher が 500本以上出してるかどうか
    '''

    def __init__(self, train: bool = True) -> None:
        super().__init__(name='pub_count_500', train=train)

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

        transform = BasicGroupByTransform(keys=['Publisher'], targets=['id'], aggs=['count'])
        pub_count = transform(df_whole)
        df_main = pd.merge(df_main, pub_count, how='left', on='Publisher')
        df_main['count_over_500_Publisher'] = df_main.loc[:, pub_count.columns[-1]] >= 500
        df_main['count_over_100_Publisher'] = df_main.loc[:, pub_count.columns[-1]] >= 100

        return df_main.loc[:, ['count_over_500_Publisher', 'count_over_100_Publisher']]
