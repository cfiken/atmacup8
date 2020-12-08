from typing import Optional, Dict
import pandas as pd

from mykaggle.feature.base import Feature
from mykaggle.lib.pandas_util import change_column_name

DATE_TE_COLUMNS = ['target_5days_mean']


class DateTE(Feature):
    '''
    日付の Target Encoding
    '''

    def __init__(self, train: bool) -> None:
        super().__init__('date_te', train)

    def create(
        self,
        base: pd.DataFrame,
        others: Optional[Dict[str, pd.DataFrame]] = None,
        *args, **kwargs
    ) -> pd.DataFrame:
        if others is None:
            raise ValueError('others 必要よ')
        df_main = others['main'].copy().loc[:, ['request_id', 'imp_at']]
        df_main['imp_day'] = pd.to_datetime(df_main.loc[:, 'imp_at']).dt.day
        if self.train:
            df = others['main'].copy().loc[:, ['request_id', 'imp_at', 'target']]
        else:
            df = others['another'].copy().loc[:, ['request_id', 'imp_at', 'target']]
        df_target_per_day = self._get_mean_target_per_day(df)
        df_main = pd.merge(df_main, df_target_per_day, how='left', on='imp_day')
        df_main = df_main.loc[:, ['request_id'] + DATE_TE_COLUMNS]

        return df_main

    def _get_mean_target_per_day(self, df: pd.DataFrame):
        df['imp_day'] = pd.to_datetime(df.loc[:, 'imp_at']).dt.day
        df_target_per_day = df.groupby('imp_day')['target'].mean().reset_index()
        df_target_per_day = change_column_name(df_target_per_day, 'target', 'day_avg_target')
        df_tmp = pd.concat([df_target_per_day, df_target_per_day], axis=0)
        df_tmp['target_5days_mean'] = df_tmp.rolling(5, center=True)['day_avg_target'].mean()
        df_target_per_day = pd.merge(df_target_per_day, df_tmp.iloc[14:44], how='left', on='imp_day')
        return df_target_per_day
