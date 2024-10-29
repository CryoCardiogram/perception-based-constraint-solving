"""
This file contains implementation of evaluations method for calibration described in
    Kull, M., Perello Nieto, M., Kängsepp, M., Silva Filho, T., Song, H., & Flach, P. (2019).
    Beyond temperature scaling: Obtaining well-calibrated multi-class probabilities with dirichlet calibration.
    Advances in neural information processing systems, 32.

The code is adapated from https://github.com/dirichletcal/experiments_dnn/blob/master/scripts/utility/evaluation.py
"""

# Functions for calibration of results
# from __future__ import division, print_function
import pickle

import numpy as np
import sklearn.metrics as metrics
from scipy.stats import percentileofscore
from sklearn.metrics import log_loss
from sklearn.preprocessing import OneHotEncoder, label_binarize


def evaluate(probs, y_true, verbose=False, normalize=True, bins=15):
    """
    Evaluate model using various scoring measures: Error Rate, ECE, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, NLL, Brier Score

    Params:
        probs: a list containing probabilities for all the classes with a shape of (samples, classes)
        y_true: a list containing the actual class labels
        verbose: (bool) are the scores printed out. (default = False)
        normalize: (bool) in case of 1-vs-K calibration, the probabilities need to be normalized.
        bins: (int) - into how many bins are probabilities divided (default = 15)

    Returns:
        (error, ece, mce, loss, brier), returns various scoring measures
    """

    preds = np.argmax(probs, axis=1)  # Take maximum confidence as prediction
    accuracy = metrics.accuracy_score(y_true, preds) * 100
    error = 100 - accuracy

    # Calculate ECE and ECE2, + Classwise and Full (ECE2 =? Full_ECE)
    ece = ECE(probs, y_true, bin_size=1 / bins)
    ece2 = ECE(probs, y_true, bin_size=1 / bins, ece_full=True, normalize=normalize)
    ece_cw = classwise_ECE(probs, y_true, bins=bins, power=1)
    ece_full = full_ECE(probs, y_true, bins=bins, power=1)

    ece_cw2 = classwise_ECE(probs, y_true, bins=bins, power=2)
    ece_full2 = full_ECE(probs, y_true, bins=bins, power=2)

    # Calculate MCE
    mce = MCE(probs, y_true, bin_size=1 / bins, normalize=normalize)
    mce2 = MCE(probs, y_true, bin_size=1 / bins, ece_full=True, normalize=normalize)

    loss = log_loss(y_true=y_true, y_pred=probs)

    # y_prob_true = np.array([probs[i, idx] for i, idx in enumerate(y_true)])  # Probability of positive class
    # brier = brier_score_loss(y_true=y_true, y_prob=y_prob_true)  # Brier Score (MSE), NB! not correct

    brier = Brier(probs, y_true)

    if verbose:
        print("Accuracy:", accuracy)
        print("Error:", error)
        print("ECE:", ece)
        print("ECE2:", ece2)
        print("ECE_CW", ece_cw)
        print("ECE_CW2", ece_cw)
        print("ECE_FULL", ece_full)
        print("ECE_FULL2", ece_full2)
        print("MCE:", mce)
        print("MCE2:", mce2)
        print("Loss:", loss)
        print("brier:", brier)

    return (
        error,
        ece,
        ece2,
        ece_cw,
        ece_cw2,
        ece_full,
        ece_full2,
        mce,
        mce2,
        loss,
        brier,
    )


def evaluate_rip(probs, y_true, verbose=False, normalize=True, bins=15):
    """
    Evaluate model using various scoring measures: Error Rate, ECE, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, NLL, Brier Score

    Params:
        probs: a list containing probabilities for all the classes with a shape of (samples, classes)
        y_true: a list containing the actual class labels
        verbose: (bool) are the scores printed out. (default = False)
        normalize: (bool) in case of 1-vs-K calibration, the probabilities need to be normalized.
        bins: (int) - into how many bins are probabilities divided (default = 15)

    Returns:
        (error, ece, mce, loss, brier), returns various scoring measures
    """

    preds = np.argmax(probs, axis=1)  # Take maximum confidence as prediction
    accuracy = metrics.accuracy_score(y_true, preds) * 100
    error = 100 - accuracy

    # Calculate ECE and ECE2, + Classwise and Full (ECE2 =? Full_ECE)
    ece = ECE(probs, y_true, bin_size=1 / bins)
    ece2 = -1
    ece_cw = classwise_ECE(probs, y_true, bins=bins, power=1)
    ece_full = -1

    ece_cw2 = -1
    ece_full2 = -1

    # Calculate MCE
    mce = MCE(probs, y_true, bin_size=1 / bins, normalize=normalize)
    mce2 = -1

    loss = log_loss(y_true=y_true, y_pred=probs)

    # y_prob_true = np.array([probs[i, idx] for i, idx in enumerate(y_true)])  # Probability of positive class
    # brier = brier_score_loss(y_true=y_true, y_prob=y_prob_true)  # Brier Score (MSE), NB! not correct

    brier = Brier(probs, y_true)

    if verbose:
        print("Accuracy:", accuracy)
        print("Error:", error)
        print("ECE:", ece)
        print("ECE2:", ece2)
        print("ECE_CW", ece_cw)
        print("ECE_CW2", ece_cw)
        print("ECE_FULL", ece_full)
        print("ECE_FULL2", ece_full2)
        print("MCE:", mce)
        print("MCE2:", mce2)
        print("Loss:", loss)
        print("brier:", brier)

    return (
        error,
        ece,
        ece2,
        ece_cw,
        ece_cw2,
        ece_full,
        ece_full2,
        mce,
        mce2,
        loss,
        brier,
    )


def evaluate_slim(probs, y_true, verbose=False, normalize=True, bins=15):
    """
    Evaluate model using various scoring measures: Error Rate, ECE, ece2, ece_cw, ece_cw2, ece_full, ece_full2, mce, mce2, NLL, Brier Score

    Params:
        probs: a list containing probabilities for all the classes with a shape of (samples, classes)
        y_true: a list containing the actual class labels
        verbose: (bool) are the scores printed out. (default = False)
        normalize: (bool) in case of 1-vs-K calibration, the probabilities need to be normalized.
        bins: (int) - into how many bins are probabilities divided (default = 15)

    Returns:
        (error, ece, mce, loss, brier), returns various scoring measures
    """

    # preds = np.argmax(probs, axis=-1)  # Take maximum confidence as prediction
    # accuracy = metrics.accuracy_score(y_true, preds) * 100
    # error = 100 - accuracy

    # Calculate ECE and ECE2, + Classwise and Full (ECE2 =? Full_ECE)
    ece = ECE(probs, y_true, bin_size=1 / bins)
    ece_cw = classwise_ECE(probs, y_true, bins=bins, power=1)

    # Calculate MCE
    mce = MCE(probs, y_true, bin_size=1 / bins, normalize=normalize)

    # loss = log_loss(y_true=y_true, y_pred=probs)

    # y_prob_true = np.array([probs[i, idx] for i, idx in enumerate(y_true)])  # Probability of positive class
    # brier = brier_score_loss(y_true=y_true, y_prob=y_prob_true)  # Brier Score (MSE), NB! not correct

    brier = Brier(probs, y_true)

    if verbose:
        print("ECE:", ece)
        print("ECE_CW", ece_cw)
        print("MCE:", mce)
        # print("Loss:", loss)
        print("brier:", brier)

    return (None, ece, ece_cw, mce, None, brier)


def softmax(x):
    """
    Compute softmax values for each sets of scores in x.

    Parameters:
        x (numpy.ndarray): array containing m samples with n-dimensions (m,n)
    Returns:
        x_softmax (numpy.ndarray) softmaxed values for initial (m,n) array
    """
    e_x = np.exp(
        x - np.max(x)
    )  # Subtract max, so the biggest is 0 to avoid numerical instability

    # Axis 0 if only one dimensional array
    axis = 0 if len(e_x.shape) == 1 else 1

    return e_x / e_x.sum(axis=axis, keepdims=1)


def get_preds_all(y_probs, y_true, axis=1, normalize=False, flatten=True):
    # Take maximum confidence as prediction
    y_preds = np.argmax(y_probs, axis=axis)
    y_preds = y_preds.reshape(-1, 1)

    if normalize:
        y_probs /= np.sum(y_probs, axis=axis).reshape(-1, 1)

    enc = OneHotEncoder(handle_unknown="ignore", sparse=False)
    enc.fit(y_preds)

    y_preds = enc.transform(y_preds)
    y_true = enc.transform(y_true)

    if flatten:
        y_preds = y_preds.flatten()
        y_true = y_true.flatten()
        y_probs = y_probs.flatten()

    return y_preds, y_probs, y_true


def compute_acc_bin_legacy(conf_thresh_lower, conf_thresh_upper, conf, pred, true):
    """
    # Computes accuracy and average confidence for bin

    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels

    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [
        x
        for x in zip(pred, true, conf)
        if x[2] > conf_thresh_lower and x[2] <= conf_thresh_upper
    ]
    if len(filtered_tuples) < 1:
        return 0, 0, 0
    else:
        # How many correct labels
        correct = len([x for x in filtered_tuples if x[0] == x[1]])
        # How many elements falls into given bin
        len_bin = len(filtered_tuples)
        avg_conf = (
            sum([x[2] for x in filtered_tuples]) / len_bin
        )  # Avg confidence of BIN
        accuracy = float(correct) / len_bin  # accuracy of BIN
        return accuracy, avg_conf, len_bin


def compute_acc_bin(
    conf_thresh_lower, conf_thresh_upper, conf, pred, true, ece_full=False
):
    """
    # Computes accuracy and average confidence for bin

    Args:
        conf_thresh_lower (float): Lower Threshold of confidence interval
        conf_thresh_upper (float): Upper Threshold of confidence interval
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        pred_thresh (float) : float in range (0,1), indicating the prediction threshold

    Returns:
        (accuracy, avg_conf, len_bin): accuracy of bin, confidence of bin and number of elements in bin.
    """
    filtered_tuples = [
        x
        for x in zip(pred, true, conf)
        if (x[2] > conf_thresh_lower or conf_thresh_lower == 0)
        and x[2] <= conf_thresh_upper
    ]

    if len(filtered_tuples) < 1:
        return 0, 0, 0
    else:
        if ece_full:
            # How many elements falls into given bin
            len_bin = len(filtered_tuples)
            avg_conf = (
                sum([x[2] for x in filtered_tuples]) / len_bin
            )  # Avg confidence of BIN
            # Mean difference from actual class
            accuracy = np.mean([x[1] for x in filtered_tuples])

        else:
            # How many correct labels
            correct = len([x for x in filtered_tuples if x[0] == x[1]])
            # How many elements falls into given bin
            len_bin = len(filtered_tuples)
            avg_conf = (
                sum([x[2] for x in filtered_tuples]) / len_bin
            )  # Avg confidence of BIN
            accuracy = float(correct) / len_bin  # accuracy of BIN

    return accuracy, avg_conf, len_bin


def ECE(probs, true, bin_size=0.1, ece_full=False, normalize=False):
    """
    Expected Calibration Error

    Args:
        probs (numpy.ndarray): list of probabilities (samples, nr_classes)
        true (numpy.ndarray): list of true labels (samples, 1)
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?

    Returns:
        ece: expected calibration error
    """

    probs = np.array(probs)
    true = np.array(true)

    if len(true.shape) == 2 and true.shape[1] > 1:
        true = true.argmax(axis=1).reshape(-1, 1)

    if ece_full:
        pred, conf, true = get_preds_all(
            probs, true, normalize=normalize, flatten=ece_full
        )
    else:
        # Take maximum confidence as prediction
        pred = np.argmax(probs, axis=1)

        if normalize:
            conf = np.max(probs, axis=1) / np.sum(probs, axis=1)
            # Check if everything below or equal to 1?
        else:
            conf = np.max(probs, axis=1)  # Take only maximum confidence

    # get predictions, confidences and true labels for all classes

    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)  # Get bounds of bins

    n = len(conf)
    ece = 0  # Starting error

    for (
        conf_thresh
    ) in upper_bounds:  # Go through bounds and find accuracies and confidences
        acc, avg_conf, len_bin = compute_acc_bin(
            conf_thresh - bin_size, conf_thresh, conf, pred, true, ece_full
        )
        ece += np.abs(acc - avg_conf) * len_bin / n  # Add weigthed difference to ECE

    return ece


def MCE(probs, true, bin_size=0.1, ece_full=False, normalize=False):
    """
    Maximal Calibration Error

    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?

    Returns:
        mce: maximum calibration error
    """

    if ece_full:
        pred, conf, true = get_preds_all(
            probs, true, normalize=normalize, flatten=ece_full
        )
    else:
        # Take maximum confidence as prediction
        pred = np.argmax(probs, axis=1)

        if normalize:
            conf = np.max(probs, axis=1) / np.sum(probs, axis=1)
            # Check if everything below or equal to 1?
        else:
            conf = np.max(probs, axis=1)  # Take only maximum confidence

    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)

    cal_errors = []

    for conf_thresh in upper_bounds:
        acc, avg_conf, _ = compute_acc_bin(
            conf_thresh - bin_size, conf_thresh, conf, pred, true, ece_full
        )
        cal_errors.append(np.abs(acc - avg_conf))

    return max(cal_errors)


def Brier(probs, true):
    """
    Brier score (mean squared error)

    Args:
        probs (list): 2-D list of probabilities
        true (list): 1-D list of true labels

    Returns:
        brier: brier score
    """

    assert len(probs) == len(true)

    n = len(true)  # number of samples
    k = len(probs[0])  # number of classes

    brier = 0

    for i in range(n):  # Go through all the samples
        for j in range(k):  # Go through all the classes
            y = 1 if j == true[i] else 0  # Check if correct class
            brier += (probs[i][j] - y) ** 2  # squared error

    # Mean squared error (should also normalize by number of classes?)
    return brier / n / k


def get_bin_info(conf, pred, true, bin_size=0.1):
    """
    Get accuracy, confidence and elements in bin information for all the bins.

    Args:
        conf (numpy.ndarray): list of confidences
        pred (numpy.ndarray): list of predictions
        true (numpy.ndarray): list of true labels
        bin_size: (float): size of one bin (0,1)  # TODO should convert to number of bins?

    Returns:
        (acc, conf, len_bins): tuple containing all the necessary info for reliability diagrams.
    """

    upper_bounds = np.arange(bin_size, 1 + bin_size, bin_size)

    accuracies = []
    confidences = []
    bin_lengths = []

    for conf_thresh in upper_bounds:
        acc, avg_conf, len_bin = compute_acc_bin(
            conf_thresh - bin_size, conf_thresh, conf, pred, true
        )
        accuracies.append(acc)
        confidences.append(avg_conf)
        bin_lengths.append(len_bin)

    return accuracies, confidences, bin_lengths


def binary_ECE(probs, y_true, power=1, bins=15):
    idx = np.digitize(probs, np.linspace(0, 1, bins)) - 1

    def bin_func(p, y, idx):
        return (
            (np.abs(np.mean(p[idx]) - np.mean(y[idx])) ** power)
            * np.sum(idx)
            / len(probs)
        )

    ece = 0
    for i in np.unique(idx):
        ece += bin_func(probs, y_true, idx == i)
    return ece


def classwise_ECE(probs, y_true, power=1, bins=15):
    probs = np.array(probs)
    if not np.array_equal(probs.shape, y_true.shape):
        y_true = label_binarize(np.array(y_true), classes=range(probs.shape[1]))

    n_classes = probs.shape[1]

    return np.mean(
        [
            binary_ECE(probs[:, c], y_true[:, c].astype(float), power=power, bins=bins)
            for c in range(n_classes)
        ]
    )


def simplex_binning(probs, y_true, bins=15):
    probs = np.array(probs)
    if not np.array_equal(probs.shape, y_true.shape):
        y_true = label_binarize(np.array(y_true), classes=range(probs.shape[1]))

    idx = np.digitize(probs, np.linspace(0, 1, bins)) - 1

    prob_bins = {}
    label_bins = {}

    for i, row in enumerate(idx):
        try:
            prob_bins[",".join([str(r) for r in row])].append(probs[i])
            label_bins[",".join([str(r) for r in row])].append(y_true[i])
        except KeyError:
            prob_bins[",".join([str(r) for r in row])] = [probs[i]]
            label_bins[",".join([str(r) for r in row])] = [y_true[i]]

    bins = []
    for key in prob_bins:
        bins.append(
            [
                len(prob_bins[key]),
                np.mean(np.array(prob_bins[key]), axis=0),
                np.mean(np.array(label_bins[key]), axis=0),
            ]
        )

    return bins


def full_ECE(probs, y_true, bins=15, power=1):
    n = len(probs)

    probs = np.array(probs)
    if not np.array_equal(probs.shape, y_true.shape):
        y_true = label_binarize(np.array(y_true), classes=range(probs.shape[1]))

    idx = np.digitize(probs, np.linspace(0, 1, bins)) - 1

    filled_bins = np.unique(idx, axis=0)

    s = 0
    for bin in filled_bins:
        i = np.where((idx == bin).all(axis=1))[0]
        s += (len(i) / n) * (
            np.abs(np.mean(probs[i], axis=0) - np.mean(y_true[i], axis=0)) ** power
        ).sum()

    return s


def label_resampling(probs):
    c = probs.cumsum(axis=1)
    u = np.random.rand(len(c), 1)
    choices = (u < c).argmax(axis=1)
    y = np.zeros_like(probs)
    y[range(len(probs)), choices] = 1
    return y


def score_sampling(probs, samples=10000, ece_function=None):
    probs = np.array(probs)

    return np.array(
        [ece_function(probs, label_resampling(probs)) for sample in range(samples)]
    )


def pECE(probs, y_true, samples=10000, ece_function=full_ECE):
    probs = np.array(probs)
    if not np.array_equal(probs.shape, y_true.shape):
        y_true = label_binarize(np.array(y_true), classes=range(probs.shape[1]))

    return 1 - (
        percentileofscore(
            score_sampling(probs, samples=samples, ece_function=ece_function),
            ece_function(probs, y_true),
        )
        / 100
    )
