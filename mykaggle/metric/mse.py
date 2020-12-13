from typing import Tuple
import numpy as np
import lightgbm as lgb
from sklearn.metrics import mean_squared_log_error, mean_squared_error


def rmsle(y_pred: np.ndarray, data: lgb.Dataset) -> Tuple[str, float, bool]:
    y_pred = np.expm1(y_pred)
    y_true = np.expm1(data.get_label())
    score = mean_squared_log_error(y_true, y_pred) ** 0.5
    return "rmsle", score, False  # name, result, is_higher_better


def rmse(y_pred: np.ndarray, data: lgb.Dataset) -> Tuple[str, float, bool]:
    score = mean_squared_error(data.get_label(), y_pred) ** 0.5
    return "rmsle", score, False  # name, result, is_higher_better


def rmse2(y_pred, data: xgb.DMatrix) -> Tuple[str, float, bool]:
    score = mean_squared_error(data.get_label(), y_pred) ** 0.5
    return "rmsle", score
