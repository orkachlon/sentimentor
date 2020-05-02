import os
import pickle
import numpy as np
import gensim.downloader as api

from abc import ABC, abstractmethod
from typing import List, Iterable, Union
from nltk import pos_tag
from nltk.corpus import stopwords as stp
from gensim.models.word2vec import Word2Vec
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from ReviewGenerator import ReviewGenerator

WORD_SPACE_DIM = 300
CUSTOM_STOP_WORDS = ['really', 'watch', 'money', 'make', 'people', 'ever', 'seen',
                     'time', 'get', 'would', 'special', 'effects', 'many']
STOP_WORDS = stp.words('english') + \
             [w.replace("'", "") for w in stp.words('english') if "'" in w] + \
             CUSTOM_STOP_WORDS


class Vectorizer(ABC):
    """
    Abstract vectorizer class
    """

    @abstractmethod
    def train(self, text: Iterable[str]) -> np.ndarray:
        """
        Trains the model on given text
        :param text: to train on
        :return: transformed text
        """
        pass

    @abstractmethod
    def transform(self, text: Iterable[str]) -> np.ndarray:
        """
        Transforms the given text without training on it
        :param text: to be transforms
        :return: transformed text
        """
        pass

    @abstractmethod
    def get_feature_names(self) -> List[str]:
        """
        :return: list of this vectorizer's features
        """
        pass


class W2v(Vectorizer):
    """
    A facade class for gensim's Word2Vec
    """

    def __init__(self, path='', load=False, local=True, limit: int = None):
        """
        Initializes a W2v with given parameters
        :param path: if load is true then path is the path from which to load the vectorizer, otherwise it is
        the path to the data on which to instantiate a ReviewGenerator
        :param load: whether to load a pre-trained vectorizer or not
        :param local: used only if load is true. Specifies whether the loaded model is from a local path or
        from gensim.api
        :param limit: used only if load is false. Specifies how many reviews to train on.
        """
        if load:
            self.vec = Word2Vec.load(os.path.join("../assets/vectorizers", path)) if local else api.load(path)
        elif len(path):
            sentences = ReviewGenerator(path, limit=limit)
            self.vec = Word2Vec(sentences=sentences, size=WORD_SPACE_DIM, workers=4)
        else:
            self.vec = Word2Vec(size=WORD_SPACE_DIM, workers=4)
        self.wv = self.vec.wv

    @staticmethod
    def _text_as_list(text: Iterable[str], flatten=False) -> Union[List[str], List[List[str]]]:
        """
        :param text: to iterate over
        :param flatten: whether to return one list with all words
        :return: the given text as a flat list of words or list of lists of words
        """
        return [w for t in text for w in t.split(' ')] if flatten else [t.split(' ') for t in text]

    def train(self, text: Iterable[str]) -> np.ndarray:
        """
        Trains the model on given text
        :param text: to train on
        :return: transformed text
        """
        flattened = W2v._text_as_list(text, True)
        self.vec.train(flattened)
        return self.transform(text)

    def save(self, name: str):
        """
        Saves this model in the 'textVectorizationModels' directory
        :param name: name to save the model by
        """
        save_path = os.path.join('../textVectorizationModels', f"w2v_{name}.model")
        print(f"Saving to {save_path} ...")
        self.vec.save(save_path)

    def transform(self, text: Iterable[str]):
        """
        Transforms the given text without training on it
        :param text: to be transforms
        :return: transformed text
        """
        text_as_lists = W2v._text_as_list(text)
        try:
            return np.array([np.array([self.wv[word] for word in t]).mean(axis=0) for t in text_as_lists])
        except KeyError:
            flattened = [w for t in text_as_lists for w in t]  # avoid calling split(' ')
            self.vec.build_vocab(flattened, update=True)
            self.vec.train(flattened, total_examples=self.vec.corpus_count, epochs=self.vec.epochs)
        return np.array([np.array([self.wv[word] for word in t]).mean(axis=0) for t in text_as_lists])

    def get_feature_names(self) -> List[str]:
        """
        :return: a list of all of this vectorizer's learned features
        """
        return self.wv.vocab.keys()


class T2v(Vectorizer):
    """
    A facade class for sklearn's TfidfVectorizer/CountVectorizer
    """

    def __init__(self, how='tfidf', name: str = None):
        """
        Initializes this T2v instance
        :param how: either 'tfidf' or 'count'. Used only if no name was given. Whether to use tfidf or bow
        :param name: if given, loads a saved T2v named 'name'
        """
        if len(name):
            self.vec = None
            self.load(name)
        elif how not in ['tfidf', 'count']:
            print(f"Method '{how}' is unrecognized, no vectorizer was made")
            self.vec = None
        else:
            vocab = set()
            with open('../assets/vocab/pos.txt', 'r') as pos:
                for w in pos:
                    vocab.add(w.strip())
            with open('../assets/vocab/neg.txt', 'r') as neg:
                for w in neg:
                    vocab.add(w.strip())
            if how == 'tfidf':
                self.vec = TfidfVectorizer(**{'sublinear_tf': True, 'ngram_range': (1, 2),
                                              'stop_words': STOP_WORDS, 'min_df': 300, 'max_df': .3,
                                              'vocabulary': vocab})
            if how == 'count':
                self.vec = CountVectorizer(**{'ngram_range': (1, 2),
                                              'stop_words': STOP_WORDS, 'min_df': .1, 'max_df': .4})

    def train(self, text: Iterable[str]) -> np.ndarray:
        """
        Trains this vectorizer on the given texts
        :param text: to train on
        :return: transformed text
        """
        if self.vec is not None:
            return self.vec.fit_transform(text).toarray()

    def transform(self, text: Iterable[str]) -> np.ndarray:
        """
        Transforms the given text without training on it
        :param text: to be transforms
        :return: transformed text
        """
        if self.vec is not None:
            return self.vec.transform(text).toarray()

    def get_feature_names(self) -> List[str]:
        """
        :return: a list of this vectorizer's features
        """
        return self.vec.get_feature_names()

    def save(self, name: str) -> None:
        """
        Saves this model to the assets directory
        :param name: of the model
        """
        with open(os.path.join("../assets/vectorizers", name), 'wb') as f:
            pickle.dump(self.vec, f)
            f.close()

    def load(self, name: str) -> None:
        """
        Loads a previously saved model from the assets directory into this vectorizer
        :param name: of the model
        """
        with open(os.path.join("../assets/vectorizers", name), 'rb') as f:
            self.vec = pickle.load(f)
            f.close()

    def score(self, X_test: np.ndarray, y_test: np.ndarray, top_n=5) -> None:
        """
        Prints the top n features of this vectorizer per label in y_test over the samples X_test
        :param X_test: samples of transformed text
        :param y_test: corresponding labels
        :param top_n: how many features to print
        """
        tagged_feats = pos_tag(self.get_feature_names())
        print(f"vector size: {len(tagged_feats)}")
        for s in sorted(np.unique(y_test)):  # 1 2 3 4 5 / pos neg
            print(f"# {'neg' if s == -1 else 'pos'}")
            # group by reviews with y == i
            # calculate feature y using its mean/chi2 on the test set
            scores = np.mean(X_test[y_test == s], axis=0)
            # scores = chi2(X_test, y_test == s)[0]
            # sort by the scores in descending order
            sorting = np.argsort(scores)[::-1]
            # get top n uni-grams and bi-grams
            uni_top = [
                          tagged_feats[i][0] for i in sorting
                          if tagged_feats[i][1] == 'JJ' and
                          len(tagged_feats[i][0].split(' ')) == 1
                       ][: top_n]
            bi_top = [
                         tagged_feats[i][0] for i in sorting
                         if tagged_feats[i][1] == 'JJ' and
                         len(tagged_feats[i][0].split(' ')) == 2
                     ][: top_n]
            all_top = [
                          tagged_feats[i][0] for i in sorting
                          if tagged_feats[i][1] == 'JJ'
                       ][: top_n]
            if np.any(list(map(lambda x: x not in uni_top, all_top))):
                print("top all: {}".format(all_top))
            print("top features: {}".format(uni_top))
            if len(bi_top):
                print("top bi: {}".format(bi_top))
            # print all bi-gram features
        print("\nall bi-grams:")
        print([feat[0] for feat in tagged_feats if len(feat[0].split(' ')) == 2])
