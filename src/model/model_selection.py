import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection._split import _BaseKFold, _num_samples, indexable
from sklearn.utils.validation import _deprecate_positional_args


class ShufflableGroupKFold(KFold):
    """
    GroupKFold with random shuffle with a sklearn-like structure
    """

    def __init__(self, n_splits: int = 5, shuffle: bool = True, random_state: int = 42):
        super().__init__(n_splits, shuffle, random_state)

    def get_n_splits(self, X: pd.DataFrame = None, y: pd.Series = None, group=None):
        return self.n_splits

    def split(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series):
        unique_ids = groups.unique()
        for fold, (tr_group_idx, va_group_idx) in enumerate(super().split(unique_ids)):
            # split group
            tr_group, va_group = unique_ids[tr_group_idx], unique_ids[va_group_idx]
            train_idx = np.where(groups.isin(tr_group))[0]
            val_idx = np.where(groups.isin(va_group))[0]
            yield train_idx, val_idx


# modified code for group gaps; source
# https://github.com/getgaurav2/scikit-learn/blob/d4a3af5cc9da3a76f0266932644b884c99724c57/sklearn/model_selection/_split.py#L2243
class PurgedGroupTimeSeriesSplit(_BaseKFold):
    """Time Series cross-validator variant with non-overlapping groups.
    Allows for a gap in groups to avoid potentially leaking info from
    train into test if the model has windowed or lag features.
    Provides train/test indices to split time series data samples
    that are observed at fixed time intervals according to a
    third-party provided group.
    In each split, test indices must be higher than before, and thus shuffling
    in cross validator is inappropriate.
    This cross-validation object is a variation of :class:`KFold`.
    In the kth split, it returns first k folds as train set and the
    (k+1)th fold as test set.
    The same group will not appear in two different folds (the number of
    distinct groups has to be at least equal to the number of folds).
    Note that unlike standard cross-validation methods, successive
    training sets are supersets of those that come before them.
    Read more in the :ref:`User Guide <cross_validation>`.
    Parameters
    ----------
    n_splits : int, default=5
        Number of splits. Must be at least 2.
    max_train_group_size : int, default=Inf
        Maximum group size for a single training set.
    group_gap : int, default=None
        Gap between train and test
    max_test_group_size : int, default=Inf
        We discard this number of groups from the end of each train split
    """

    @_deprecate_positional_args
    def __init__(
        self,
        n_splits=5,
        *,
        max_train_group_size=np.inf,
        max_test_group_size=np.inf,
        group_gap=None,
        verbose=False
    ):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.max_train_group_size = max_train_group_size
        self.group_gap = group_gap
        self.max_test_group_size = max_test_group_size
        self.verbose = verbose

    def split(self, X, y=None, groups=None):
        """Generate indices to split data into training and test set.
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : array-like of shape (n_samples,)
            Always ignored, exists for compatibility.
        groups : array-like of shape (n_samples,)
            Group labels for the samples used while splitting the dataset into
            train/test set.
        Yields
        ------
        train : ndarray
            The training set indices for that split.
        test : ndarray
            The testing set indices for that split.
        """
        if groups is None:
            raise ValueError("The 'groups' parameter should not be None")

        X, y, groups = indexable(X, y, groups)
        n_samples = _num_samples(X)

        n_splits = self.n_splits
        group_gap = self.group_gap
        max_test_group_size = self.max_test_group_size
        max_train_group_size = self.max_train_group_size

        n_folds = n_splits + 1
        group_dict = {}

        u, ind = np.unique(groups, return_index=True)
        unique_groups = u[np.argsort(ind)]
        n_samples = _num_samples(X)
        n_groups = _num_samples(unique_groups)

        for idx in np.arange(n_samples):
            if groups[idx] in group_dict:
                group_dict[groups[idx]].append(idx)

            else:
                group_dict[groups[idx]] = [idx]

        if n_folds > n_groups:
            raise ValueError(
                (
                    "Cannot have number of folds={0} greater than"
                    " the number of groups={1}"
                ).format(n_folds, n_groups)
            )

        group_test_size = min(n_groups // n_folds, max_test_group_size)
        group_test_starts = range(
            n_groups - n_splits * group_test_size, n_groups, group_test_size
        )
        for group_test_start in group_test_starts:
            train_array = []
            test_array = []

            group_st = max(0, group_test_start - group_gap - max_train_group_size)

            for train_group_idx in unique_groups[
                group_st : (group_test_start - group_gap)
            ]:
                train_array_tmp = group_dict[train_group_idx]

                train_array = np.sort(
                    np.unique(
                        np.concatenate((train_array, train_array_tmp)), axis=None
                    ),
                    axis=None,
                )

            # train_end = train_array.size

            for test_group_idx in unique_groups[
                group_test_start : group_test_start + group_test_size
            ]:
                test_array_tmp = group_dict[test_group_idx]
                test_array = np.sort(
                    np.unique(np.concatenate((test_array, test_array_tmp)), axis=None),
                    axis=None,
                )

            test_array = test_array[group_gap:]

            if self.verbose > 0:
                pass

            yield [int(i) for i in train_array], [int(i) for i in test_array]
