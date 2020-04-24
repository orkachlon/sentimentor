import os
import nltk
import pickle
import random
import operator
import numpy as np
import pandas as pd
import seaborn as sns

from typing import *
from statistics import mode
from textblob import TextBlob
from matplotlib import pyplot as plt
from flair.data import Sentence
from flair.models import TextClassifier
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.vader import SentiText, SentimentIntensityAnalyzer  # accuracy on bin_train.csv: 0.692
from nltk.sentiment.sentiment_analyzer import SentimentAnalyzer
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from dev.textVectorization.text_vectorization import T2v
from dev.ReviewGenerator import BalancedReviewGenerator, BinaryReviewGenerator

MODELS_DIR = 'sentimentClassificationModels'
NUM_MODELS = 6
NUM_STARS = 5


def try_resampled_data(df: pd.DataFrame, resample_target: int, test_model=True):
    # resample data
    resampled_df = balance(df, resample_target)
    # train vectorizer on training data
    vectorizer = T2v()
    X_train, y_train, X_test, y_test = split_data(resampled_df, .75)
    X_train = vectorizer.train(X_train)
    X_test = vectorizer.transform(X_test)
    # train scaler on training data and transform it and the test data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train, y_train)
    scaler.transform(X_test, False)
    # train logistic regression model on training data
    resampled_logreg = LogisticRegression(max_iter=5000).fit(X_train, y_train)
    # test the model on test data
    if test_model:
        log_classifier(resampled_logreg, X_test, y_test, f"Resampled to {resample_target} logistic regression")
    return resampled_logreg


def try_penalized_svm(df: pd.DataFrame, scale=True, test_model=True) -> SVC:
    vectorizer = T2v()
    X_train, y_train, X_test, y_test = split_data(df, .75)
    X_train = vectorizer.train(X_train)
    X_test = vectorizer.transform(X_test)
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train, y_train)
        scaler.transform(X_test)
    svm = SVC(kernel='linear', class_weight='balanced', probability=True)
    svm.fit(X_train, y_train)
    if test_model:
        log_classifier(svm, X_test, y_test, "penalized SVM")
    return svm


def try_random_forest(df: pd.DataFrame, scale=True, test_model=True) -> RandomForestClassifier:
    vectorizer = T2v()
    X_train, y_train, X_test, y_test = split_data(df, .75)
    vectorizer.train(df.text)
    X_train = vectorizer.transform(X_train)
    X_test = vectorizer.transform(X_test)
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train, y_train)
        scaler.transform(X_test, False)
    classifier = RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0)
    classifier.fit(X_train, y_train)
    if test_model:
        log_classifier(classifier, X_test, y_test, 'RandomForestClassifier')
    return classifier


def try_kNN(df: pd.DataFrame, k: Union[int, range] = 5, scale=True, test_model=True) -> KNeighborsClassifier:
    # train vectorizer on all data
    vec = T2v()
    vec.train(df.text)
    # split data and transform with vectorizer
    X_train, y_train, X_test, y_test = split_data(df, .75)
    X_train = vec.transform(X_train)
    X_test = vec.transform(X_test)
    if scale:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)
    knn = None
    if type(k) == int:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)
        if test_model:
            log_classifier(knn, X_test, y_test, f"kNN, k = {k}")
    elif type(k) == range:
        scores_list = []
        for i in k:
            knn = KNeighborsClassifier(n_neighbors=i)
            knn.fit(X_train, y_train)
            if test_model:
                y_pred = knn.predict(X_test)
                scores_list.append(accuracy_score(y_test, y_pred))
        if test_model:
            plt.plot(k, scores_list)
            plt.xlabel('k')
            plt.ylabel('accuracy')
            plt.show()
    return knn


def test_pretrained_model(df: pd.DataFrame, name: str) -> None:
    correct = ((df.score == 'pos') & (df.pred > 0)) | ((df.score == 'neg') & (df.pred < 0))
    print(f"{name} accuracy: {len(df.loc[correct]) / df.shape[0]}")


def try_vader(df: pd.DataFrame, test_model=True) -> SentimentIntensityAnalyzer:
    sid = SentimentIntensityAnalyzer()
    df['pred'] = list(map(lambda s: sid.polarity_scores(s)['compound'], df.text))
    if test_model:
        test_pretrained_model(df, 'VADER')
    return sid


def try_textblob(df: pd.DataFrame, test_model=True) -> None:
    df['pred'] = list(map(lambda x: TextBlob(x).sentiment.polarity, df.text))
    if test_model:
        test_pretrained_model(df, 'Textblob')


def try_flair(df: pd.DataFrame, test_model=True) -> None:
    model = TextClassifier.load('en-sentiment')  # type: TextClassifier
    df['sentence_obj'] = list(map(Sentence, df.text))
    model.predict(df.sentence_obj.to_list())
    df['pred'] = list(map(lambda s: 1 if s.labels[0].value == 'POSITIVE' else - 1, df.sentence_obj))
    if test_model:
        test_pretrained_model(df, 'Flair')


def try_combined(df: pd.DataFrame, test_model=True) -> None:
    vader = SentimentIntensityAnalyzer()
    fl = TextClassifier.load('en-sentiment')  # type: TextClassifier
    df['sentence_obj'] = list(map(Sentence, df.text))
    fl.predict(df.sentence_obj.to_list())
    df['flair_pred'] = list(map(lambda s: s.labels[0].score * (1 if s.labels[0].value == 'POSITIVE' else -1),
                                df['sentence_obj']))

    df['textblob_pred'] = list(map(lambda x: TextBlob(x).sentiment.polarity, df.text))

    df['vader_pred'] = list(map(lambda s: vader.polarity_scores(s)['compound'], df.text))
    df['pred'] = df[['vader_pred', 'textblob_pred', 'flair_pred']] @ np.array([.6, .6, .9])
    if test_model:
        test_pretrained_model(df, "Combined vader and flair")


def model_selection(df: pd.DataFrame, vec) -> None:
    X = df.text
    y = df.score
    X = vec.train(X)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X, y)
    models_with_scaling = [
        SVC(kernel='linear'),
        LogisticRegression(random_state=0, max_iter=1000)
    ]
    models_without_scaling = [
        # RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
        # MultinomialNB()
    ]
    CV = 5
    cv_df = pd.DataFrame(index=range(CV * (len(models_with_scaling) + len(models_without_scaling))))
    entries = []
    for model in models_with_scaling:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, X_scaled, y, scoring='accuracy', cv=CV)
        for fold_idx, accuracy, in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))

    for model in models_without_scaling:
        model_name = model.__class__.__name__
        accuracies = cross_val_score(model, X, y, scoring='accuracy', cv=CV)
        for fold_idx, accuracy, in enumerate(accuracies):
            entries.append((model_name, fold_idx, accuracy))

    cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
    sns.boxplot(x='model_name', y='accuracy', data=cv_df)
    sns.stripplot(x='model_name', y='accuracy', data=cv_df, size=8, jitter=True, edgecolor='gray', linewidth=2)
    plt.show()


def log_classifier(classifier, X_true, y_true, name: str = None) -> None:
    y_pred = classifier.predict(X_true)
    if name is not None:
        print(f"{name} details:")
    print(f"accuracy: {accuracy_score(y_true, y_pred)}")
    print(f"predicted classes: {np.unique(y_pred)}")


def balance(df: pd.DataFrame, size: int, log=True):
    dfs = []
    for i in range(NUM_STARS):
        if df[df.score == i + 1].shape[0] != size:
            dfs.append(resample(df[df.score == i + 1], replace=True, n_samples=size, random_state=123))
        else:
            dfs.append(df[df.score == i + 1])
    combined = pd.concat(dfs)
    if log:
        print(f"balance:\n{combined.score.value_counts()}")
    return combined


def plot_data_balance(df: pd.DataFrame) -> None:
    fig = plt.figure(figsize=(8, 6))
    df.groupby('score').text.count().plot.bar(ylim=0)
    plt.show()


def split_data(df: pd.DataFrame, ratio: float = .75, log=True):
    sep = int(df.shape[0] * ratio)
    X_train = df.text[: sep]
    X_test = df.text[sep:]
    y_train, y_test = df.score[: sep], df.score[sep:]
    if log:
        print(f"=== split ===\ntrain: {len(X_train)}\ntest: {len(X_test)}")
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    # sample balanced data
    # df = pd.concat([pd.DataFrame(BalancedReviewGenerator(f"movies_{i}", 1000, True)) for i in range(1, 11)])
    df = BinaryReviewGenerator("../ml-dataset/bin_train.csv", lim=10).as_df()
    print(f"balance:\n{df.groupby('score').count()}")
    # print(df.head())

    # default Logistic regression without taking data imbalance into account
    # scaler = StandardScaler()
    # vectorizer = T2v()
    # X_train, y_train, X_test, y_test = split_data(df, vectorizer)
    # X_train = scaler.fit_transform(X_train, y_train)
    # scaler.transform(X_test, False)
    # logreg = LogisticRegression(max_iter=5000).fit(X_train, y_train)
    # log_classifier(logreg, X_test, y_test, "default logistic regression")

    # =========== SOLUTIONS ===========

    # 1. Upsample minorities
    # upsampled_logreg = try_resampled_data(df, np.max(df.score.value_counts()))

    # 2. Downsample majorities
    # downsampled_logreg = try_resampled_data(df, np.min(df.score.value_counts()))

    # 3. Penalized SVM
    # try_penalized_svm(df)

    # 4. Tree based algorithm
    # try_random_forest(df)

    # ===========================
    # = classifiers from link 3 =
    # ===========================

    # model_selection(df, T2v())

    # ===================
    # = kNN from link 6 =
    # ===================

    # try_kNN(df, range(1, 26))

    # ===========================
    # = classifiers from link 7 =
    # ===========================

    # try_vader(df)
    # try_textblob(df)

    # sample less ! this take a lot of time !
    # try_flair(df)

    try_combined(df)
