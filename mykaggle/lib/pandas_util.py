from typing import Union, List
import pandas as pd


def change_column_name(df: Union[pd.DataFrame, pd.Series],
                       old: Union[str, List[str]],
                       new: Union[str, List[str]]) -> pd.DataFrame:
    if isinstance(df, pd.Series):
        df = df.to_frame()
    if not isinstance(old, list) and not isinstance(new, list):
        return df.rename(columns={old: new})

    name_map = {}
    for o, n in zip(old, new):
        name_map[o] = n
    return df.rename(columns=name_map)
