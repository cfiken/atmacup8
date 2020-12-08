from typing import Optional, Dict
import pandas as pd
from sklearn.decomposition import PCA

from mykaggle.feature.base import Feature
from mykaggle.transform.pivot import PivotTransform

COLUMNS = [
    'Genre',
    'Platform',
    'Year_of_Release'
]


class PubToCategoryPivotPCAAll(Feature):
    '''
    Publisher から見た各カテゴリ pivot and pca
    '''

    def __init__(self, train: bool = True, n_components: int = 3) -> None:
        super().__init__(name='pub_to_category_pivot_pca_all', train=train)
        self.n_components = n_components

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

        df_pivot = None
        for i, c in enumerate(COLUMNS):
            transform = PivotTransform(indices=['Publisher'], column=c, target='id', aggs=['count'], fillna=0)
            pub_to_c = transform(df_whole)
            if df_pivot is None:
                df_pivot = pub_to_c
            else:
                df_pivot = pd.merge(df_pivot, pub_to_c, how='left', on='Publisher')

        df_pivot = df_pivot.fillna(0)
        df_pca = pd.DataFrame(self._pca_transform(df_pivot, self.n_components))

        df_pivot = pd.concat([df_pivot, df_pca], axis=1)
        df_pivot = df_pivot.iloc[:, [0] + list(range(-1, -self.n_components - 1, -1))]
        pca_columns = ['_'.join(['pca', str(n), 'count_id_pivotby_Publisher_for_all'])
                       for n in range(self.n_components)]
        df_pivot.columns = ['Publisher'] + pca_columns
        df_main = pd.merge(df_main, df_pivot, how='left', on='Publisher')
        return df_main.loc[:, pca_columns]

    def _pca_transform(self, df: pd.DataFrame, n_components: int):
        pca = PCA(n_components)
        return pca.fit_transform(df.drop('Publisher', axis=1).values)
