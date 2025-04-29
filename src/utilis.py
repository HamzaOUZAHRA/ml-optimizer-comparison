import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


def load_processed_data(base_dir=None):
    """
    Load train/test splits from the processed data directory.

    Parameters
    ----------
    base_dir : str, optional
        Root of the project (where 'data/processed' resides). Defaults to parent of this file.

    Returns
    -------
    X_train, y_train, X_test, y_test : np.ndarray
    """
    if base_dir is None:
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
    proc_dir = os.path.join(base_dir, 'data', 'processed')
    X_train = np.load(os.path.join(proc_dir, 'X_train.npy'))
    y_train = np.load(os.path.join(proc_dir, 'y_train.npy'))
    X_test  = np.load(os.path.join(proc_dir, 'X_test.npy'))
    y_test  = np.load(os.path.join(proc_dir, 'y_test.npy'))
    return X_train, y_train, X_test, y_test


def evaluate_model(name, y_true, y_pred, y_prob=None):
    """
    Compute common classification metrics and print them.

    Parameters
    ----------
    name : str
        Name of the model/optimizer.
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_prob : array-like, optional
        Probabilities for positive class, for ROC AUC.

    Returns
    -------
    metrics : dict
        Dictionary of metrics.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    results = {
        'accuracy': acc,
        'precision': prec,
        'recall': rec,
        'f1_score': f1,
    }
    print(f"{name} -- Acc: {acc:.4f}, Prec: {prec:.4f}, Rec: {rec:.4f}, F1: {f1:.4f}")
    if y_prob is not None:
        auc = roc_auc_score(y_true, y_prob)
        results['roc_auc'] = auc
        print(f"{name} -- ROC AUC: {auc:.4f}")
    return results


def plot_loss_histories(histories, labels, title, xlabel='Iteration', ylabel='Loss', figsize=(8,4)):
    """
    Plot multiple loss histories on the same axes.

    Parameters
    ----------
    histories : list of array-like
        Loss values per iteration for each curve.
    labels : list of str
        Labels for each curve.
    title : str
        Plot title.
    xlabel : str
    ylabel : str
    figsize : tuple
    """
    plt.figure(figsize=figsize)
    for hist, lbl in zip(histories, labels):
        plt.plot(hist, label=lbl)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()


def plot_loss_vs_time(time_histories, loss_histories, labels, title, xlabel='Time (s)', ylabel='Loss', figsize=(8,4)):
    """
    Plot loss vs time for multiple curves.

    Parameters
    ----------
    time_histories : list of array-like
        Cumulative time stamps for each iteration.
    loss_histories : list of array-like
        Loss values per iteration for each curve.
    labels : list of str
        Labels for each curve.
    title : str
        Plot title.
    xlabel : str
    ylabel : str
    figsize : tuple
    """
    plt.figure(figsize=figsize)
    for t_h, l_h, lbl in zip(time_histories, loss_histories, labels):
        plt.plot(t_h, l_h, label=lbl)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
