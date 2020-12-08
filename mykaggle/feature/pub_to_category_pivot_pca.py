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


class PubToCategoryPivotPCA(Feature):
    '''
    Publisher から見た各カテゴリ pivot and pca
    '''

    def __init__(self, train: bool = True, n_components: int = 3) -> None:
        super().__init__(name='pub_to_category_pivot_pca', train=train)
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

        all_columns = []
        for c in COLUMNS:
            transform = PivotTransform(indices=['Publisher'], column=c, target='id', aggs=['count'], fillna=0)
            pub_to_c = transform(df_whole)
            df_pca = pd.DataFrame(self._pca_transform(pub_to_c, self.n_components))
            pub_to_c = pd.concat([pub_to_c, df_pca], axis=1)
            pub_to_c = pub_to_c.iloc[:, [0] + list(range(-1, -self.n_components - 1, -1))]
            pca_columns = ['_'.join(['pca', str(n), 'count_id_pivotby_Publisher_for', c])
                           for n in range(self.n_components)]
            all_columns.extend(pca_columns)
            pub_to_c.columns = ['Publisher'] + pca_columns
            df_main = pd.merge(df_main, pub_to_c, how='left', on='Publisher')
        return df_main.loc[:, all_columns]

    def _pca_transform(self, df: pd.DataFrame, n_components: int):
        pca = PCA(n_components)
        return pca.fit_transform(df.drop('Publisher', axis=1).values)
