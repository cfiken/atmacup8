from typing import Optional, Dict
import pandas as pd
from mykaggle.feature.base import Feature


class ScoreTBD(Feature):
    '''
    UserScore が TBD かどうか
    '''

    def __init__(self, train: bool = True) -> None:
        super().__init__(name='score_tbd', train=train)

    def create(
        self,
        base: pd.DataFrame,
        others: Optional[Dict[str, pd.DataFrame]] = None,
        *args, **kwargs
    ) -> pd.DataFrame:
        df_main = others['main'].copy()
        user_score = df_main.loc[:, 'User_Score']
        df_main['is_tbd'] = False
        df_main.loc[user_score == 'tbd', 'is_tbd'] = True
        return df_main.loc[:, ['is_tbd']]
