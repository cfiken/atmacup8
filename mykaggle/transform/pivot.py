from typing import Any, Union, List, Optional
import pandas as pd

from mykaggle.transform.base import BaseTransform


class PivotTransform(BaseTransform):
    '''
    指定された indices, columns, aggs で pivot して返す。
    '''

    def __init__(
        self,
        indices: List[str],
        column: str,
        target: str,
        aggs: List[str],
        fillna: Optional[Any] = None
    ) -> None:
        '''
        Args:
          index: 集約キー
          columns: pivot 後の column とするキー、これごとに集約できる
          target: 集約対象のキー
          aggs: 集約後に行う演算
          fillna: nan を埋める場合、埋める値
        '''
        self.indices = indices
        self.column = column
        self.target = target
        self.aggs = aggs
        self.fillna = fillna

    def __call__(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.transform(df)

    def fit(self, X: pd.DataFrame) -> 'PivotTransform':
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.pivot(df, self.indices, self.column, self.target, self.aggs)

    def pivot(
        self,
        df: pd.DataFrame,
        indices: List[str],
        column: str,
        target: str,
        aggs: List[str],
        *args, **kwargs
    ) -> pd.DataFrame:
        df_output = df.pivot_table(index=indices, columns=column, values=target, aggfunc=aggs).reset_index()
        column_values = [c[1] for c in df_output.columns[len(indices):]]
        new_columns = indices + self._prepare_pivoted_columns(indices, column, column_values, target, aggs)
        df_output.columns = new_columns
        if self.fillna is not None:
            df_output = df_output.fillna(self.fillna)
        return df_output

    def _prepare_pivoted_columns(
        self,
        indices: List[str],
        column: str,
        column_values: List[str],
        target: str,
        aggs: List[Union[str, Any]]
    ) -> List[str]:
        aggs = [a if isinstance(a, str) else a.__name__ for a in aggs]
        return [
            '_'.join([a, target, 'pivotby', index, 'for', column, str(c)])
            for c in column_values
            for index in indices
            for a in aggs
        ]
