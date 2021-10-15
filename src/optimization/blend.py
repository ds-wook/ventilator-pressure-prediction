from typing import List

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import GroupKFold

from utils.utils import LoggerFactory

logger = LoggerFactory().getLogger(__name__)


def get_score(
    weights: np.ndarray, train_idx: List[int], oofs: List[np.ndarray], labels
) -> float:
    blending = np.zeros_like(oofs[0][train_idx, :])

    for oof, weight in zip(oofs[:-1], weights):
        blending += weight * oof[train_idx, :]

    blending += (1 - np.sum(weights)) * oofs[-1][train_idx, :]
    return mean_absolute_error(labels[train_idx, :], blending)


def get_best_weights(oofs: np.ndarray, groups: pd.Series, labels: np.ndarray) -> float:
    weight_list = []
    weights = np.array([1 / len(oofs) for x in range(len(oofs) - 1)])

    kf = GroupKFold(n_splits=5)
    splits = kf.split(oofs[0], labels, groups)

    for fold, (train_idx, valid_idx) in enumerate(splits):
        res = minimize(
            get_score,
            weights,
            args=(train_idx, oofs, labels),
            method="Nelder-Mead",
            tol=1e-6,
        )
        logger.info(f"fold: {fold} res.x: {res.x}")
        weight_list.append(res.x)

    mean_weight = np.mean(weight_list, axis=0)
    print(f"optimized weight: {mean_weight}")
    return mean_weight
