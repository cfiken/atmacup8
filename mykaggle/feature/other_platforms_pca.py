from typing import Optional, Dict
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder
from mykaggle.feature.base import Feature
from mykaggle.lib.pandas_util import change_column_name


class OtherPlatformsPCA(Feature):
    '''
    自分以外で発売されてる Platform の onehot encoding
    '''

    def __init__(self, train: bool = True) -> None:
        super().__init__(name='other_platforms_pca', train=train)

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

        le = LabelEncoder()
        df_whole['le_Platform'] = le.fit_transform(df_whole.loc[:, 'Platform'])
        df_main['le_Platform'] = le.transform(df_main.loc[:, 'Platform'])
        platform_list = df_whole.groupby('Name')['le_Platform'].apply(list).reset_index()
        platform_list = change_column_name(platform_list, 'le_Platform', 'other_platforms')
        df_main = pd.merge(df_main, platform_list, how='left', on='Name')

        # nan 処理
        def _nan_platforms(x: pd.DataFrame):
            if np.isnan(x['other_platforms']).any():
                return []
            return x['other_platforms']

        df_main['other_platforms'] = df_main[['other_platforms', 'le_Platform']].apply(_nan_platforms, axis=1)
        num_platforms = len(df_whole['le_Platform'].unique())
        df_pca_op = self._onehot_platforms_pca(df_main.loc[:, 'other_platforms'], num_platforms, df_main.shape[0])
        df_main = pd.concat([df_main, df_pca_op], axis=1)
        return df_main.loc[:, ['pca_0_other_platforms', 'pca_1_other_platforms']]

    def _onehot_platforms_pca(self, other_platforms: pd.Series, num_platforms: int, num_data: int) -> pd.DataFrame:
        onehot = np.zeros((num_data, num_platforms), dtype=np.int32)
        for i, other_platform in enumerate(other_platforms):
            for p in other_platform:
                onehot[i, p] = 1
        pca_op = self._pca_transform(onehot)
        df_pca = pd.DataFrame(pca_op)
        df_pca.columns = ['pca_0_other_platforms', 'pca_1_other_platforms']
        return df_pca

    def _pca_transform(self, values: np.ndarray, n_components: int = 2):
        pca = PCA(n_components)
        return pca.fit_transform(values)
