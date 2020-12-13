from typing import Optional, Dict
import pandas as pd
from mykaggle.feature.base import Feature


class JPSales(Feature):
    '''
    元データの has_xx_sales, データが追加された country_prob のデータが必要
    '''

    def __init__(self, train: bool = True) -> None:
        super().__init__(name='jp_sales', train=train)

    def create(
        self,
        base: pd.DataFrame,
        others: Optional[Dict[str, pd.DataFrame]] = None,
        *args, **kwargs
    ) -> pd.DataFrame:
        df_main = others['main'].copy()
        if self.train:
            df_jp = pd.read_csv('../ckpt/248_jp_sales/train_248_jp_sales.csv')
        else:
            df_jp = pd.read_csv('../ckpt/248_jp_sales/248_jp_sales.csv')
        df_main['JP_Sales'] = df_jp['JP_Sales']
        return df_main.loc[:, ['JP_Sales']]
