from typing import Optional, Dict
import pandas as pd
import numpy as np

from mykaggle.lib.pandas_util import change_column_name
from mykaggle.feature.base import Feature

COLUMNS = [
    'Critic_Score', 'User_Score', 'Critic_Count', 'User_Count'
]


class ScoreReconstruct(Feature):
    '''
    Critic_Score, User_Score, Critic_Count, User_Count
    '''

    def __init__(self, train: bool = True) -> None:
        '''
        :params name: 特徴の名前 e.g.) gender, age
        :params train: train 用の特徴であれば True, test 用であれば False
        :params category: 特徴をまとめる dir を作る場合指定する e.g.) 特定コンペの名前など
        '''
        super().__init__(name='score_reconstruct', train=train)

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

        df_main.loc[df_main.loc[:, 'User_Score'] == 'tbd', 'User_Score'] = np.nan
        df_whole.loc[df_whole.loc[:, 'User_Score'] == 'tbd', 'User_Score'] = np.nan

        # normalize
        df_main.loc[:, 'User_Score'] = df_main.loc[:, 'User_Score'].astype(np.float32) / 10.0
        df_main.loc[:, 'Critic_Score'] = df_main.loc[:, 'Critic_Score'] / 100.0
        df_whole.loc[:, 'User_Score'] = df_whole.loc[:, 'User_Score'].astype(np.float32) / 10.0
        df_whole.loc[:, 'Critic_Score'] = df_whole.loc[:, 'Critic_Score'] / 100.0

        df_main = self._merge_agg_scores(df_whole, df_main)

        def user_score_reconstruct(x):
            diff = x['Critic_Score'] - x['mean_critic_score_by_platform']
            normalized_diff = diff / x['std_critic_score_by_platform']
            reconstructed = x['mean_user_score_by_platform'] + normalized_diff * x['std_user_score_by_platform']
            return reconstructed

        def critic_score_reconstruct(x):
            normalized_diff = (x['User_Score'] - x['mean_user_score_by_platform']) / x['std_user_score_by_platform']
            reconstructed = x['mean_critic_score_by_platform'] + normalized_diff * x['std_critic_score_by_platform']
            return reconstructed

        us_notna_cr_na = ~df_main['User_Score'].isna() & df_main['Critic_Score'].isna()
        us_na_cr_notna = ~df_main['Critic_Score'].isna() & df_main['User_Score'].isna()

        df_main.loc[us_na_cr_notna, 'User_Score'] = df_main.loc[us_na_cr_notna].apply(user_score_reconstruct, axis=1)
        df_main.loc[us_na_cr_notna, 'User_Count'] = df_main.loc[us_na_cr_notna, 'mean_user_count_by_platform']
        df_main.loc[us_notna_cr_na, 'Critic_Score'] = df_main.loc[us_notna_cr_na].apply(
            critic_score_reconstruct, axis=1)
        df_main.loc[us_notna_cr_na, 'Critic_Count'] = df_main.loc[us_notna_cr_na, 'mean_critic_count_by_platform']

        return df_main.loc[:, COLUMNS]

    def _merge_agg_scores(self, df: pd.DataFrame, merge_to_df: pd.DataFrame) -> pd.DataFrame:
        user_score_agg = df.groupby(['Platform'])['User_Score'].agg(['mean', 'std'])
        user_score_agg = change_column_name(
            user_score_agg, ['mean', 'std'], ['mean_user_score_by_platform', 'std_user_score_by_platform'])
        critic_score_agg = df.groupby(['Platform'])['Critic_Score'].agg(['mean', 'std'])
        critic_score_agg = change_column_name(
            critic_score_agg, ['mean', 'std'], ['mean_critic_score_by_platform', 'std_critic_score_by_platform'])

        user_count_agg = df.groupby(['Platform'])['User_Count'].agg('mean')
        user_count_agg = change_column_name(user_count_agg, 'User_Count', 'mean_user_count_by_platform')
        critic_count_agg = df.groupby(['Platform'])['Critic_Count'].agg('mean')
        critic_count_agg = change_column_name(critic_count_agg, 'Critic_Count', 'mean_critic_count_by_platform')

        merge_to_df = pd.merge(merge_to_df, user_score_agg, how='left', on='Platform')
        merge_to_df = pd.merge(merge_to_df, critic_score_agg, how='left', on='Platform')
        merge_to_df = pd.merge(merge_to_df, user_count_agg, how='left', on='Platform')
        merge_to_df = pd.merge(merge_to_df, critic_count_agg, how='left', on='Platform')

        return merge_to_df
