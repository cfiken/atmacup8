from typing import Optional, Dict
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from mykaggle.feature.base import Feature
from mykaggle.transform.pivot import PivotTransform

COLUMNS = [
    'Genre',
    'Platform',
    'Year_of_Release',
    'Developer'
]


class PubToCategoryPivotKMeans(Feature):
    '''
    Publisher から見た各カテゴリ pivot and pca
    '''

    def __init__(self, train: bool = True, n_components: int = 3, n_clusters: int = 8) -> None:
        super().__init__(name='pub_to_category_pivot_kmeans', train=train)
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

        df_pivot = df_pivot.fillna(0)
        pca = self._pca_transform(df_pivot, self.n_components)
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=1019)
        clusters = kmeans.fit_predict(pca)
        column_name = 'kmeans_cluster_by_Publisher_pivotby_all'
        df_pivot[column_name] = clusters
        df_main = pd.merge(df_main, df_pivot, how='left', on='Publisher')
        # transform = BasicGroupByTransform(keys=[column_name], targets=['target'], aggs=['mean'])
        # cluster_target = transform(df_main)
        # df_main = pd.merge(df_main, cluster_target, how='left', on=column_name)
        return df_main.loc[:, [column_name]]

    def _pca_transform(self, df: pd.DataFrame, n_components: int):
        pca = PCA(n_components)
        return pca.fit_transform(df.drop('Publisher', axis=1).values)
