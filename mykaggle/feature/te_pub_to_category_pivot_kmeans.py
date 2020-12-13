from typing import Optional, Dict
import pandas as pd
from sklearn.decomposition import PCA

from mykaggle.lib.pandas_util import change_column_name
from mykaggle.feature.base import Feature
from mykaggle.transform.pivot import PivotTransform
from mykaggle.transform.groupby import BasicGroupByTransform

COLUMNS = [
    'Genre',
    'Platform',
    'Year_of_Release',
    'Developer'
]


class TEPubToCategoryPivotKMeans(Feature):
    '''
    Publisher から見た各カテゴリ pivot and pca
    '''

    def __init__(self, train: bool = True, n_components: int = 3, n_clusters: int = 8) -> None:
        super().__init__(name='te_pub_to_category_pivot_kmeans', train=train)
        self.n_components = n_components
        self.n_clusters = n_clusters

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

        column_name = 'kmeans_cluster_by_Publisher_pivotby_all'
        df_main[column_name] = base.copy().loc[:, column_name]
        df_main = change_column_name(df_main, column_name, 'kmeans_cluster')
        transform = BasicGroupByTransform(keys=['kmeans_cluster'], targets=['target'], aggs=['mean'])
        cluster_target = transform(df_main)
        df_main = pd.merge(df_main, cluster_target, how='left', on='kmeans_cluster')
        return df_main.loc[:, [cluster_target.columns[-1]]]

    def _pca_transform(self, df: pd.DataFrame, n_components: int):
        pca = PCA(n_components)
        return pca.fit_transform(df.drop('Publisher', axis=1).values)
