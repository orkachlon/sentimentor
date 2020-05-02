import numpy as np
import pandas as pd
import seaborn as sns

from typing import Union, List, Iterable, Tuple
from textblob import TextBlob
from matplotlib import pyplot as plt
from flair.data import Sentence
from flair.models import TextClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from nltk.sentiment.vader import SentimentIntensityAnalyzer


class CombinedClassifier:
    """
    A pre-trained text classifier built from a combination of nltk's vader,
    flair's TextClassifier and Textblob
    """

    vader: SentimentIntensityAnalyzer
    flair: TextClassifier
    weights: np.ndarray

    def __init__(self, weights: np.ndarray = np.array([.6, .6, .9]) / 2.1):
        """
        Initializes the classifiers this combined classifier is built on
        :param weights: how much weight to give each classifier when predicting
        """
        self.vader = SentimentIntensityAnalyzer()
        self.flair = TextClassifier.load('en-sentiment')
        self.weights = weights

    def predict(self, X: Iterable[str]) -> Union[np.ndarray, pd.Series]:
        """
        Predicts the scores of the given sample set
        :param X: the sample set
        :return: prediction over given sample set
        """
        as_sentence_obj = list(map(Sentence, X))
        self.flair.predict(as_sentence_obj)
        fl_pred = list(map(lambda s: s.labels[0].score * (1 if s.labels[0].value == 'POSITIVE' else -1),
                           as_sentence_obj))
        tb_pred = list(map(lambda x: TextBlob(x).sentiment.polarity, X))
        vd_pred = list(map(lambda s: self.vader.polarity_scores(s)['compound'], X))
        return np.vstack((fl_pred, tb_pred, vd_pred)).T @ self.weights

    def score(self, X: Iterable[str], y: Union[np.ndarray, pd.Series]) -> dict:
        """
        :param X: the samples - iterable over string
        :param y: the true labels of the sample set - an array of the scores in {-1, +1}
        :return: dictionary with various stats about the model regarding the given test set
        """
        y_pred = np.sign(self.predict(X))
        score_dict = {
            'num_samples': len(y),
            'TP': np.count_nonzero(np.logical_and(y_pred == 1, y_pred == y)),
            'FP': np.count_nonzero(np.logical_and(y_pred == 1, y_pred != y)),
            'TN': np.count_nonzero(np.logical_and(y_pred == -1, y_pred == y)),
            'FN': np.count_nonzero(np.logical_and(y_pred == -1, y_pred != y))
        }
        P = score_dict['TP'] + score_dict['FP']
        N = score_dict['TN'] + score_dict['FN']
        score_dict['error'] = (score_dict['FP'] + score_dict['FN']) / (P + N) if P + N else 0
        score_dict['accuracy'] = (score_dict['TP'] + score_dict['TN']) / (P + N) if P + N else 0
        score_dict['FPR'] = score_dict['FP'] / N if N else 0
        score_dict['TPR'] = score_dict['TP'] / P if P else 0
        score_dict['precision'] = score_dict['TP'] / (score_dict['TP'] + score_dict['FP']) if P else 0
        score_dict['recall'] = score_dict['TP'] / P if P else 0
        return score_dict


def log_classifier(classifier,
                   X_test: Union[List[str], pd.Series],
                   y_true: Union[np.ndarray, pd.Series],
                   name: str = None) -> None:
    """
    Calculates and prints the accuracy of classifier on X_test with the true labels y_true
    :param classifier:
    :param X_test:
    :param y_true:
    :param name:
    :return:
    """
    y_pred = classifier.predict(X_test)
    if name is not None:
        print(f"{name} details:")
    print(f"accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"predicted classes: {np.unique(y_pred)}")


def plot_confusion(classifier: CombinedClassifier,
                   X: Iterable[str],
                   y: Union[np.ndarray, pd.Series]) -> None:
    """
    Plots the confusion matrix of the given classifier on the given test set
    :param classifier: TrainedTextualClassifier instance
    :param X: test set
    :param y: test labels
    """
    score = classifier.score(X, y)
    df_cm = pd.DataFrame(np.array([[score['TP'], score['FP']],
                                   [score['FN'], score['TN']]]) /
                         (score['TP'] + score['FP'] + score['TN'] + score['FN']),
                         index=['pos', 'neg'], columns=['pos', 'neg'])
    plt.figure()
    sns.heatmap(df_cm, annot=True, center=.5)
    plt.show()


def balance(df: pd.DataFrame, size: int, log=True) -> pd.DataFrame:
    """
    Resamples the data so that each label has 'size' samples
    :param df: DataFrame holding the data
    :param size: desired amount of samples for each label
    :param log: whether to print the result or not
    :return: DataFrame with resampled data
    """
    dfs = []
    for i in sorted(np.unique(df.score)):
        if df[df.score == i].shape[0] != size:
            dfs.append(resample(df[df.score == i], replace=True, n_samples=size, random_state=123))
        else:
            dfs.append(df[df.score == i])
    combined = pd.concat(dfs)
    if log:
        print(f"balance:\n{combined.score.value_counts()}")
    return combined


def plot_data_balance(df: pd.DataFrame) -> None:
    """
    Plots the data balance for each label
    :param df: holding the data
    """
    plt.figure(figsize=(8, 6))
    df.groupby('y').text.count().plot.bar(ylim=0)
    plt.show()


def split_data(df: pd.DataFrame, ratio: float = .75, log=True) -> Tuple[Iterable[str],
                                                                        Iterable[str],
                                                                        Union[np.ndarray, pd.Series],
                                                                        Union[np.ndarray, pd.Series]]:
    """
    Splits data into train set and test set
    :param df: DataFrame with columns ['text', 'score']
    :param ratio: how much of the data to use for training
    :param log: print the split or not
    :return: tuple: (train set, train labels, test set, test labels)
    """
    sep = int(df.shape[0] * ratio)
    X_train = df.text[: sep]
    X_test = df.text[sep:]
    y_train, y_test = df.score[: sep], df.score[sep:]
    if log:
        print(f"=== split ===\ntrain: {len(X_train)}\ntest: {len(X_test)}")
    return X_train, y_train, X_test, y_test
