from typing import Optional, Dict
import pandas as pd
from mykaggle.feature.base import Feature


class RegionSales(Feature):
    '''
    元データの has_xx_sales, データが追加された country_prob のデータが必要
    '''

    def __init__(self, train: bool = True) -> None:
        super().__init__(name='region_sales', train=train)

    def create(
        self,
        base: pd.DataFrame,
        others: Optional[Dict[str, pd.DataFrame]] = None,
        *args, **kwargs
    ) -> pd.DataFrame:
        df_main = others['main'].copy()
        if self.train:
            df_jp = pd.read_csv('../ckpt/248_jp_sales/train_248_jp_sales.csv')
            df_na = pd.read_csv('../ckpt/256_na_sales/train_256_na_sales.csv')
            df_eu = pd.read_csv('../ckpt/252_eu_sales/train_252_eu_sales.csv')
            df_other = pd.read_csv('../ckpt/253_others_sales/train_253_others_sales.csv')
        else:
            df_jp = pd.read_csv('../ckpt/248_jp_sales/248_jp_sales.csv')
            df_na = pd.read_csv('../ckpt/256_na_sales/256_na_sales.csv')
            df_eu = pd.read_csv('../ckpt/252_eu_sales/252_eu_sales.csv')
            df_other = pd.read_csv('../ckpt/253_others_sales/253_others_sales.csv')
        df_main['JP_Sales'] = df_jp['JP_Sales']
        df_main['NA_Sales'] = df_na['NA_Sales']
        df_main['EU_Sales'] = df_eu['EU_Sales']
        df_main['Other_Sales'] = df_other['Other_Sales']
        return df_main.loc[:, ['JP_Sales', 'NA_Sales', 'EU_Sales', 'Other_Sales']]
