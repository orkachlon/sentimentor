import os
import nltk
import random
import pickle
import numpy as np
import pandas as pd
import seaborn as sns

from statistics import mode
from matplotlib import pyplot as plt
from nltk.classify import ClassifierI
from nltk.tokenize import word_tokenize
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.utils import resample
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score

from dev.textVectorization.text_vectorization import T2v
from dev.ReviewGenerator import BalancedReviewGenerator

MODELS_DIR = 'sentimentClassificationModels'
NUM_MODELS = 6
NUM_STARS = 5


class SentimentAnalyzer:
    _NUM_FEATURES = 5000

    def __init__(self, load: bool = False, file_num: int = None, data_dir: str = '../reviewsByStarRating'):
        if load:
            self.__voted_classifier = SentimentAnalyzer._load_classifiers()
        else:
            if file_num is not None:
                self.__word_features = None
                self._train_classifiers(data_dir, file_num)

    def sentiment(self, text: str):
        features = self.find_features(text)
        return self.__voted_classifier.classify(features), self.__voted_classifier.confidence(features)

    @staticmethod
    def _load_classifiers():
        classifiers = np.empty(NUM_MODELS, dtype=np.object)
        for i, file in enumerate(os.listdir(MODELS_DIR)):
            if os.path.isfile(file):
                with open(f"{MODELS_DIR}/{file}", "rb") as f:
                    classifiers[i] = pickle.load(f)
        return VoteClassifier(*tuple(classifiers))

    def find_features(self, document):
        words = word_tokenize(document)
        features = {}
        for w in self.__word_features:
            features[w] = w in words
        return features

    @staticmethod
    def _save_classifier(classifier, filename):
        f = open(f"sentimentClassificationModels/{filename}.pickle", 'wb')
        pickle.dump(classifier, f)
        f.close()

    def _get_training_data(self, data_dir, file_num):
        wf_path = f"pickledData/word_features{int(SentimentAnalyzer._NUM_FEATURES / 1000)}k.pickle"
        doc_path = f"pickledData/documents.pickle"
        self.__word_features = []
        documents = []
        doc_load = False
        wf_load = False
        if os.path.exists(wf_path):
            wf_load = True
            self.__word_features = SentimentAnalyzer._load_pickled(wf_path)
        if os.path.exists(doc_path):
            doc_load = True
            documents = SentimentAnalyzer._load_pickled(doc_path)
        if not len(self.__word_features) or not doc_load:
            reviews = []
            for i in range(NUM_STARS):
                f = open(f"{data_dir}/movie_reviews_{file_num}_score{i + 1}.csv", 'r')
                reviews.append(pd.read_csv(f, delimiter=',', quotechar='"', escapechar='\\', header=0))
                f.close()

            all_words = []

            #  allow only adjectives
            adjectives = ['JJ']

            for i in range(NUM_STARS):
                # reviews[i].dropna(inplace=True, how='any')  # this is done in the data_organizer
                for j, rev in reviews[i].iterrows():
                    if not doc_load:
                        documents.append((rev['text'], float(i + 1)))
                    words = word_tokenize(rev['text'])
                    pos = nltk.pos_tag(words)
                    for w in pos:
                        if w[1] in adjectives:
                            all_words.append(w[0])

            if not doc_load:
                SentimentAnalyzer._pickle(documents, f"pickledData/documents.pickle")

            # Create word histogram to get most frequent words
            all_words = nltk.FreqDist(all_words)

            # Select 5000 most common words to use as features
            if not wf_load:
                self.__word_features = all_words.most_common(SentimentAnalyzer._NUM_FEATURES)
                SentimentAnalyzer._pickle(self.__word_features,
                                          f"pickledData/word_features{int(SentimentAnalyzer._NUM_FEATURES / 1000)}k.pickle")

        if os.path.exists("pickledData/feature_sets5k.pickle"):
            feature_sets = [self._load_pickled(f"pickledData/feature_sets/feature_sets5k_{i + 1}.pickle") for i in range(25000)]
        else:
            feature_sets = [(self.find_features(rev), category) for (rev, category) in documents[: 25000]]
            for i, feat in enumerate(feature_sets):
                self._pickle(feat, f"pickledData/feature_sets/feature_sets5k_{i + 1}.pickle")

        random.shuffle(feature_sets)
        sep = int(len(feature_sets) * .75)
        return feature_sets[: sep], feature_sets[sep:]

    @staticmethod
    def _pickle(obj, path):
        with open(path, 'wb') as f:
            pickle.dump(obj, f)
            f.close()

    @staticmethod
    def _load_pickled(path):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
            f.close()
        return obj

    def _train_classifiers(self, data_dir, file_num):
        training_set, testing_set = self._get_training_data(data_dir, file_num)

        if os.path.exists(f"sentimentClassificationModels/naive_bayes5k.pickle"):
            naive_bayes = self._load_pickled(f"sentimentClassificationModels/naive_bayes5k.pickle")
        else:
            naive_bayes = nltk.NaiveBayesClassifier.train(training_set)
            SentimentAnalyzer._save_classifier(naive_bayes, "naive_bayes5k")
        print("Original Naive Bayes Algo accuracy percent:",
              (nltk.classify.accuracy(naive_bayes, testing_set)) * 100)
        naive_bayes.show_most_informative_features(15)

        if os.path.exists(f"sentimentClassificationModels/mnb_classifier5k.pickle"):
            mnb_classifier = self._load_pickled(f"sentimentClassificationModels/mnb_classifier5k.pickle")
        else:
            mnb_classifier = SklearnClassifier(MultinomialNB())
            mnb_classifier.train(training_set)
            SentimentAnalyzer._save_classifier(mnb_classifier, "mnb_classifier5k")
        print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(mnb_classifier, testing_set)) * 100)

        if os.path.exists(f"sentimentClassificationModels/bernoulliNB_classifier5k.pickle"):
            bernoulliNB_classifier = self._load_pickled(
                f"sentimentClassificationModels/bernoulliNB_classifier5k.pickle")
        else:
            bernoulliNB_classifier = SklearnClassifier(BernoulliNB())
            bernoulliNB_classifier.train(training_set)
            SentimentAnalyzer._save_classifier(bernoulliNB_classifier, "bernoulliNB_classifier5k")
        print("BernoulliNB_classifier accuracy percent:",
              (nltk.classify.accuracy(bernoulliNB_classifier, testing_set)) * 100)

        if os.path.exists(f"sentimentClassificationModels/logistic_regression_classifier5k.pickle"):
            logistic_regression_classifier = self._load_pickled(
                f"sentimentClassificationModels/logistic_regression_classifier5k.pickle")
        else:
            logistic_regression_classifier = SklearnClassifier(LogisticRegression())
            logistic_regression_classifier.train(training_set)
            SentimentAnalyzer._save_classifier(logistic_regression_classifier, "logistic_regression_classifier5k")
        print("LogisticRegression_classifier accuracy percent:",
              (nltk.classify.accuracy(logistic_regression_classifier, testing_set)) * 100)

        if os.path.exists(f"sentimentClassificationModels/linearSVC_classifier5k.pickle"):
            linearSVC_classifier = self._load_pickled(
                f"sentimentClassificationModels/linearSVC_classifier5k.pickle")
        else:
            linearSVC_classifier = SklearnClassifier(LinearSVC())
            linearSVC_classifier.train(training_set)
            SentimentAnalyzer._save_classifier(linearSVC_classifier, "linearSVC_classifier5k")
        print("LinearSVC_classifier accuracy percent:",
              (nltk.classify.accuracy(linearSVC_classifier, testing_set)) * 100)

        if os.path.exists(f"sentimentClassificationModels/sgdc_classifier5k.pickle"):
            sgdc_classifier = self._load_pickled(f"sentimentClassificationModels/sgdc_classifier5k.pickle")
        else:
            sgdc_classifier = SklearnClassifier(SGDClassifier())
            sgdc_classifier.train(training_set)
            SentimentAnalyzer._save_classifier(sgdc_classifier, "sgdc_classifier5k")
        print("SGDClassifier accuracy percent:", nltk.classify.accuracy(sgdc_classifier, testing_set) * 100)

        self.__voted_classifier = VoteClassifier(naive_bayes, mnb_classifier, bernoulliNB_classifier,
                                                 logistic_regression_classifier, linearSVC_classifier,
                                                 sgdc_classifier)


class VoteClassifier(ClassifierI):

    def __init__(self, *classifiers):
        self._classifiers = classifiers

    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)

    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)

        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf

    def labels(self):
        pass


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


def try_penalized_svm(df, scale=True, test_model=True):
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


def try_random_forest(df, scale=True, test_model=True):
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


def model_selection(df, vec):
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


def log_classifier(classifier, X_true, y_true, name=None):
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


def plot_data_balance(df):
    fig = plt.figure(figsize=(8, 6))
    df.groupby('score').text.count().plot.bar(ylim=0)
    plt.show()


def split_data(df, ratio: float, log=True):
    sep = int(df.shape[0] * ratio)
    X_train = df.text[: sep]
    X_test = df.text[sep:]
    y_train, y_test = df.score[: sep], df.score[sep:]
    if log:
        print(f"=== split ===\ntrain: {len(X_train)}\ntest: {len(X_test)}")
    return X_train, y_train, X_test, y_test


if __name__ == '__main__':
    # sample balanced data
    df = pd.concat([pd.DataFrame(BalancedReviewGenerator(f"movies_{i}", 1000, True)) for i in range(1, 6)])
    print(f"balance:\n{df.groupby('score').count()}")

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

    model_selection(df, T2v('count'))
