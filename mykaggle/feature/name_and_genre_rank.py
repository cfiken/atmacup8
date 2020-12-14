from typing import Optional, Dict
import pandas as pd
from mykaggle.feature.base import Feature


class NameAndGenreRank(Feature):
    '''
    1st place solution から
    '''

    def __init__(self, train: bool = True) -> None:
        super().__init__(name='name_and_genre_rank', train=train)

    def create(
        self,
        base: pd.DataFrame,
        others: Optional[Dict[str, pd.DataFrame]] = None,
        *args, **kwargs
    ) -> pd.DataFrame:
        df_main = others['main'].copy()

        df_main_sorted_name = df_main.sort_values(['Name', 'Year_of_Release'])
        df_main_sorted_genre = df_main.sort_values(['Genre', 'Year_of_Release'])

        ny_rank = df_main_sorted_name.groupby(['Name', 'Year_of_Release']).cumcount()
        gy_rank = df_main_sorted_genre.groupby(['Genre', 'Year_of_Release']).cumcount()
        ny_count = df_main_sorted_name.groupby(['Name', 'Year_of_Release'])['Name'].transform('count')
        gy_count = df_main_sorted_genre.groupby(['Genre', 'Year_of_Release'])['Name'].transform('count')
        df_main['Name_rank_num_per'] = (ny_rank / ny_count).sort_index()
        df_main['Genre_rank_num_per'] = (gy_rank / gy_count).sort_index()

        return df_main.loc[:, ['Name_rank_num_per', 'Genre_rank_num_per']]
