from typing import List
from pathlib import Path

import lightgbm as lgb
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def compute_importances(
    importances: pd.DataFrame,
    columns: List[str],
    model: lgb.Booster,
    fold: int
) -> pd.DataFrame:
    imp_df = pd.DataFrame()
    imp_df['feature'] = columns
    imp_df['gain'] = model.feature_importance('gain')
    imp_df['fold'] = fold + 1
    importances = pd.concat([importances, imp_df], axis=0, sort=False)
    return importances


def save_importances(importances: pd.DataFrame, ckptdir: Path) -> None:
    mean_gain = importances[['gain', 'feature']].groupby('feature').mean()
    importances['mean_gain'] = importances['feature'].map(mean_gain['gain'])
    print(importances.shape)
    plt.figure(figsize=(10, 20))
    sns.barplot(
        x='gain',
        y='feature',
        data=importances.sort_values('mean_gain', ascending=False)[:500])
    plt.tight_layout()
    plt.savefig(ckptdir / 'importances.png')
