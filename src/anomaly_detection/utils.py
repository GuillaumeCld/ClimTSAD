import numpy as np
import torch
from numba import njit, float64, int64
from numba.typed import List

from .eval import Evaluator
from .training import Trainer

def burst_compute(ts, n_period, threshold, period_length, min_length, max_length, split):
    all_bursts = exhaustive_search(
        ts, n_period, threshold, period_length, min_length, max_length, split)
    return merge_bursts(all_bursts)

@njit((float64[:], int64, float64, int64, int64, int64, int64))
def exhaustive_search(
    ts: np.array,
    n_period: int,
    threshold: float,
    period_length: int,
    min_length: int,
    max_length: int,
    split: int,
):
    """
    Perform an exhaustive search for bursts in a time series.

    This function identifies seasonal bursts in a time series `ts` by calculating z-scores
    for different periods and lengths. A burst is defined as a period where the z-score
    exceeds a given threshold.

    The reference statistics (mean and standard deviation) are estimated **only** on the
    `split` first periods (the training window), while z-scores -- and therefore bursts --
    are only computed on the periods at or after `split`. This makes it possible to detect
    bursts in the recent part of the series using the behaviour observed in the past.

    Parameters:
        ts (np.array): The input time series data.
        n_period (int): The number of periods in the time series.
        threshold (float): The z-score threshold to identify bursts.
        period_length (int): The length of each period.
        min_length (int): The minimum length of a burst.
        max_length (int): The maximum length of a burst.
        split (int): Index of the first period used for detection. Periods
            [0, split) are used to estimate mean/std, periods [split, n_period)
            are scanned for bursts. Must satisfy 2 <= split < n_period.
            If you have an absolute index in `ts`, pass `index // period_length`.

    Returns:
        List: A list of tuples where each tuple contains the start index, end index,
              and z-score of a burst.
    """

    if split < 2:
        raise ValueError("split must be >= 2 to estimate an unbiased variance")
    if split >= n_period:
        raise ValueError("split must be strictly smaller than n_period")

    n = len(ts)
    v = np.copy(ts).reshape(n_period, period_length)
    all_bursts = List()

    # Finding burst of length 1 (day) only the z-score are needed
    if min_length == 1:
        inv = 1 / split
        invm1 = 1 / (split - 1)  # -1 for unbiased variance
        for i in range(period_length):
            train = v[:split, i]
            mean = np.sum(train) * inv
            squared_dev = np.sum(np.square(train - mean)) * invm1
            std = np.sqrt(squared_dev)
            if std < 1e-10:
                continue
            test = v[split:, i]
            z_score = (test - mean) / std
            flagged_years = np.where(np.abs(z_score) >= threshold)[0]
            for y in flagged_years:
                start = i + (y + split) * period_length
                all_bursts.append((start, start, z_score[y]))
        end_index = n
    else:
        size = min_length
        end_index = n - size + 1
        e_span = period_length - size + 1
        end_year = end_index // period_length
        # The last (truncated) year is never part of the training window,
        # so all training years are complete: no e_span handling needed here.
        n_train = min(split, end_year)

        v[0, 0] = np.sum(ts[:size])
        for j in range(1, period_length):
            v[0, j] = v[0, j - 1] + ts[j + size - 1] - ts[j - 1]

        mean_v = np.copy(v[0, :])
        std_v = np.zeros(period_length, dtype=ts.dtype)

        for year in range(1, end_year):
            start = year * period_length
            v[year, 0] = (
                v[year - 1, period_length - 1] + ts[start + size - 1] - ts[start - 1]
            )
            for j in range(1, period_length):
                v[year, j] = (
                    v[year, j - 1] + ts[start + j + size - 1] - ts[start + j - 1]
                )

            # welford update, training window only
            if year < n_train:
                inv = 1 / (year + 1)
                tmp = v[year, :] - mean_v
                std_v += year * inv * tmp * tmp
                mean_v += tmp * inv

        # Last year special case to handle interval out of bounds
        start = end_year * period_length
        v[end_year, 0] = (
            v[end_year - 1, period_length - 1] + ts[start + size - 1] - ts[start - 1]
        )
        for j in range(1, e_span):
            v[end_year, j] = (
                v[end_year, j - 1] + ts[start + j + size - 1] - ts[start + j - 1]
            )

        if n_train >= 2:
            std_v = np.sqrt(std_v / (n_train - 1))  # -1 for unbiased variance

            for year in range(split, end_year):
                z_score = v[year, :] - mean_v
                for date in range(period_length):
                    std = std_v[date]
                    if std >= 1e-10 and np.abs(z_score[date] / std) >= threshold:
                        start = date + year * period_length
                        all_bursts.append(
                            (start, start + size - 1, z_score[date] / std)
                        )

            if end_year >= split:
                z_score = v[end_year, :e_span] - mean_v[:e_span]
                for date in range(e_span):
                    std = std_v[date]
                    if std >= 1e-10 and np.abs(z_score[date] / std) >= threshold:
                        start = date + end_year * period_length
                        all_bursts.append(
                            (start, start + size - 1, z_score[date] / std)
                        )

    # Finding burst of length min_length+1 <= l <= max_length
    for size in range(min_length + 1, max_length + 1):
        end_index -= 1
        e_span = end_index % period_length
        end_year = end_index // period_length
        n_train = min(split, end_year)

        start = size - 1
        end = start + period_length
        v[0, :] += ts[start:end]
        mean_v = np.copy(v[0, :])
        std_v = np.zeros(period_length, dtype=ts.dtype)

        # Compute aggregates, vectorized by year
        for year in range(1, end_year):
            start = year * period_length + size - 1
            end = start + period_length
            v[year, :] += ts[start:end]
            # welford update, training window only
            if year < n_train:
                inv = 1 / (year + 1)
                tmp = v[year, :] - mean_v
                std_v += year * inv * tmp * tmp
                mean_v += tmp * inv

        # Last year special case to handle interval out of bounds
        start = end_year * period_length + size - 1
        end = start + e_span
        v[end_year, :e_span] += ts[start:end]

        if n_train < 2:
            # Not enough complete training periods for this burst length
            continue

        std_v = np.sqrt(std_v / (n_train - 1))  # -1 for unbiased variance

        for year in range(split, end_year):
            z_score = v[year, :] - mean_v
            for date in range(period_length):
                std = std_v[date]
                if std >= 1e-10 and np.abs(z_score[date] / std) >= threshold:
                    start = date + year * period_length
                    all_bursts.append((start, start + size - 1, z_score[date] / std))

        if end_year >= split:
            z_score = v[end_year, :e_span] - mean_v[:e_span]
            for date in range(e_span):
                std = std_v[date]
                if std >= 1e-10 and np.abs(z_score[date] / std) >= threshold:
                    start = date + end_year * period_length
                    all_bursts.append((start, start + size - 1, z_score[date] / std))



    return all_bursts


def merge_bursts(sequences):
    """
    Compute the Merged bursts representation i.e. the union of the given sequences with z_score of the same sign.

    Parameters:
        sequences (List): list of lists of integers (i_min, i_max, z_score)

    Returns:
        list (List): list of tuples of integers (i_min, i_max, sign(z_score))
    """

    # Separate the sequences with positive and negative z_score
    positive_intervals = [
        (interval[0], interval[1]) for interval in sequences if interval[2] > 0
    ]
    negative_intervals = [
        (interval[0], interval[1]) for interval in sequences if interval[2] < 0
    ]
    # Merge the sequences with positive and negative z_score
    positive_merged_intervals = signed_merged(positive_intervals, 1)
    negative_merged_intervals = signed_merged(negative_intervals, -1)
    # Concatenate the positive and negative sequences
    merged_intervals = positive_merged_intervals + negative_merged_intervals

    return merged_intervals


def signed_merged(intervals, sign):
    """
    Compute the union of the given intervals.

    Parameters:
        intervals (List): list of tuples of integers (i_min, i_max)
        sign (int): sign of the z_score (1 or -1)

    Returns:
        list (list): list of tuples of integers (i_min, i_max, sign)
    """
    if not intervals:
        return []
    # Sort intervals by the starting index
    intervals.sort()
    n = len(intervals)
    # Create the union of the intervals
    merged_intervals = []
    i = 0
    while i < n:
        start, end = intervals[i]

        j = i + 1
        while j < n and intervals[j][0] <= end + 1:
            end = max(end, intervals[j][1])
            j += 1

        merged_intervals.append((start, end, sign))
        i = j

    return merged_intervals




def train_and_evaluate(train,
                       test,
                       model,
                       win_size,
                       epochs,
                       trainer,
                       device=None,
                       batch_size=2048,
                       strategy="overlapping"):
    """

    """

    train = np.asarray(train)
    test = np.asarray(test)
    if train.ndim == 1:
        train = train.reshape(-1, 1)
    if test.ndim == 1:
        test = test.reshape(-1, 1)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if win_size is None:
        win_size = trainer.win_size
    else:
        trainer.win_size = win_size

    trainer.device = device

    trainer.train(model, train, epochs)
    evaluator = Evaluator(batch_size=batch_size, device=device, strategy=strategy)

    return evaluator.reconstruction_error(test, model, win_size)
