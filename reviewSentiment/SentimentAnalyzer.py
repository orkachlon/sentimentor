import os
import nltk
import random
import time
import pandas as pd
import numpy as np
# from nltk.corpus import movie_reviews
from nltk.classify.scikitlearn import SklearnClassifier
import pickle
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC
from nltk.classify import ClassifierI
from statistics import mode
from nltk.tokenize import word_tokenize

MODELS_DIR = 'sentimentClassificationModels'
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
        classifiers = [nltk.NaiveBayesClassifier, ]
        i = 0
        for file in os.listdir(MODELS_DIR):
            if os.path.isfile(file):
                with open(f"{MODELS_DIR}/{file}", "rb") as f:
                    classifiers[i] = pickle.load(f)
                i += 1
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

        feature_sets = [(self.find_features(rev), category) for (rev, category) in documents]
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

        naive_bayes = nltk.NaiveBayesClassifier.train(training_set)
        print("Original Naive Bayes Algo accuracy percent:",
              (nltk.classify.accuracy(naive_bayes, testing_set)) * 100)
        naive_bayes.show_most_informative_features(15)
        SentimentAnalyzer._save_classifier(naive_bayes, "naive_bayes5k")

        mnb_classifier = SklearnClassifier(MultinomialNB())
        mnb_classifier.train(training_set)
        print("MNB_classifier accuracy percent:", (nltk.classify.accuracy(mnb_classifier, testing_set)) * 100)
        SentimentAnalyzer._save_classifier(mnb_classifier, "mnb_classifier5k")

        bernoulliNB_classifier = SklearnClassifier(BernoulliNB())
        bernoulliNB_classifier.train(training_set)
        print("BernoulliNB_classifier accuracy percent:",
              (nltk.classify.accuracy(bernoulliNB_classifier, testing_set)) * 100)
        SentimentAnalyzer._save_classifier(bernoulliNB_classifier, "bernoulliNB_classifier5k")

        logistic_regression_classifier = SklearnClassifier(LogisticRegression())
        logistic_regression_classifier.train(training_set)
        print("LogisticRegression_classifier accuracy percent:",
              (nltk.classify.accuracy(logistic_regression_classifier, testing_set)) * 100)
        SentimentAnalyzer._save_classifier(logistic_regression_classifier, "logistic_regression_classifier5k")

        linearSVC_classifier = SklearnClassifier(LinearSVC())
        linearSVC_classifier.train(training_set)
        print("LinearSVC_classifier accuracy percent:",
              (nltk.classify.accuracy(linearSVC_classifier, testing_set)) * 100)
        SentimentAnalyzer._save_classifier(linearSVC_classifier, "linearSVC_classifier5k")

        sgdc_classifier = SklearnClassifier(SGDClassifier())
        sgdc_classifier.train(training_set)
        print("SGDClassifier accuracy percent:", nltk.classify.accuracy(sgdc_classifier, testing_set) * 100)
        SentimentAnalyzer._save_classifier(sgdc_classifier, "sgdc_classifier5k")

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


if __name__ == '__main__':
    sent = SentimentAnalyzer(file_num=1)
    with open("../reviewsByStarRating/movie_reviews_2_score5.csv", 'r') as f:
        f.readline()
        text = f.readline()
        f.close()
    start = time.time()
    classification, confidence = sent.sentiment(text)
    print(f"prediction time: {(time.time() - start):.2f} s")
    print(f"Classification: {classification}\nConfidence: {confidence}")
