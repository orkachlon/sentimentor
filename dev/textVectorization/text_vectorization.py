import os
import random
import pickle
import utils as utils
import numpy as np
import pandas as pd
import gensim.downloader as api
from abc import ABC, abstractmethod
from nltk import pos_tag
from nltk.corpus import stopwords as stp
from sklearn.feature_selection import chi2
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from ReviewGenerator import ReviewGenerator, BalancedReviewGenerator, BinaryReviewGenerator
from preprocessing.preprocessor import CSV_TO_PD_KWARGS

WORD_SPACE_DIM = 300
CUSTOM_STOP_WORDS = ['really', 'watch', 'money', 'make', 'people', 'ever', 'seen',
                     'time', 'get', 'would', 'special', 'effects', 'many']
STOP_WORDS = stp.words('english') + \
             [w.replace("'", "") for w in stp.words('english') if "'" in w] + \
             CUSTOM_STOP_WORDS


class Vectorizer(ABC):
    @abstractmethod
    def train(self, text):
        pass

    @abstractmethod
    def transform(self, text):
        pass

    @abstractmethod
    def get_feature_names(self):
        pass

    @abstractmethod
    def measure_accuracy(self, X_test, y_test, top_n: int = 5):
        pass


class W2v(Vectorizer):
    def __init__(self, path='', load=False, local=True, limit: int = None):
        if load:
            self.vec = Word2Vec.load(path) if local else api.load(path)
        elif len(path):
            sentences = ReviewGenerator(path, limit=limit) if local else api.load(path)
            self.vec = Word2Vec(sentences=sentences, size=WORD_SPACE_DIM, workers=4)
        else:
            self.vec = Word2Vec(size=WORD_SPACE_DIM, workers=4)
        self.wv = self.vec.wv

    @staticmethod
    def _text_as_list(text, flatten=False):
        return [w for t in text for w in t.split(' ')] if flatten else [t.split(' ') for t in text]

    def train(self, text):
        flattened = W2v._text_as_list(text, True)
        self.vec.train(flattened)
        return self.transform(text)

    def save(self, name):
        """
        Saves this model in the 'textVectorizationModels' directory
        :param name: name to save the model by
        """
        save_path = os.path.join('../textVectorizationModels', f"w2v_{name}.model")
        print(f"Saving to {save_path} ...")
        self.vec.save(save_path)

    def transform(self, text):
        text_as_lists = W2v._text_as_list(text)
        try:
            return np.array([np.array([self.wv[word] for word in t]).mean(axis=0) for t in text_as_lists])
        except KeyError:
            flattened = [w for t in text_as_lists for w in t]  # avoid calling split(' ')
            self.vec.build_vocab(flattened, update=True)
            self.vec.train(flattened, total_examples=self.vec.corpus_count, epochs=self.vec.epochs)
        return np.array([np.array([self.wv[word] for word in t]).mean(axis=0) for t in text_as_lists])

    def get_feature_names(self):
        return self.wv.vocab.keys()

    def measure_accuracy(self, X_test, y_test, top_n: int = 5):
        print(f"vector size: {self.vec.wv.vector_size}")
        print(self.wv.most_similar(X_test))
        # for s in sorted(np.unique(y_test)):  # 1 2 3 4 5
        #     print(f"# {s}")
        #     # group by reviews with score == i
        #     response = np.nonzero(np.array(y_test == s))[0]
        #     # calculate feature score using its mean on the test set
        #     scores = np.mean(X_test[response], axis=0)
        #     # sort by the scores in descending order
        #     sorting = np.argsort(scores)[::-1]


class T2v(Vectorizer):

    def __init__(self, how='tfidf', name=''):
        if len(name):
            self.vec = None
            self.load(name)
        elif how not in ['tfidf', 'count']:
            print(f"Method '{how}' is unrecognized, no vectorizer was made")
            self.vec = None
        else:
            vocab = set()
            with open('../ml-dataset/vocab/pos.txt', 'r') as pos:
                for w in pos:
                    vocab.add(w.strip())
            with open('../ml-dataset/vocab/neg.txt', 'r') as neg:
                for w in neg:
                    vocab.add(w.strip())
            if how == 'tfidf':
                self.vec = TfidfVectorizer(**{'sublinear_tf': True, 'ngram_range': (1, 2),
                                              'stop_words': STOP_WORDS, 'min_df': 300, 'max_df': .3,
                                              'vocabulary': vocab})
            if how == 'count':
                self.vec = CountVectorizer(**{'ngram_range': (1, 2),
                                              'stop_words': STOP_WORDS, 'min_df': .1, 'max_df': .4})

    def train(self, text):
        if self.vec is not None:
            return self.vec.fit_transform(text).toarray()

    def transform(self, text):
        if self.vec is not None:
            return self.vec.transform(text).toarray()

    def get_feature_names(self):
        return self.vec.get_feature_names()

    def save(self, name):
        with open(utils.relpath(os.path.join("textVectorization/textVectorizationModels", name)), 'wb') as f:
            pickle.dump(self.vec, f)
            f.close()

    def load(self, name):
        with open(utils.relpath(os.path.join("textVectorization/textVectorizationModels", name)), 'rb') as f:
            self.vec = pickle.load(f)
            f.close()

    def measure_accuracy(self, X_test, y_test, top_n: int = 5):
        tagged_feats = pos_tag(self.get_feature_names())
        print(f"vector size: {len(tagged_feats)}")
        for s in sorted(np.unique(y_test)):  # 1 2 3 4 5 / pos neg
            print(f"# {s}")
            # group by reviews with score == i
            # calculate feature score using its mean/chi2 on the test set
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
            print("top uni: {}".format(uni_top))
            if len(bi_top):
                print("top bi: {}".format(bi_top))
            # print all bi-gram features
        print("\nall bi-grams:")
        print([feat[0] for feat in tagged_feats if len(feat[0].split(' ')) == 2])


class D2v(Vectorizer):
    def __init__(self, path, load=False, limit=None):
        if load:
            self.vec = Doc2Vec.load(path)
        else:
            self.vec = Doc2Vec(D2v._as_tagged_doc(path, limit), vector_size=WORD_SPACE_DIM, workers=4)

    @staticmethod
    def _as_tagged_doc(path='', text=None, limit=None):
        if path:
            df = pd.DataFrame()
            if os.path.isdir(path):
                for file in os.listdir(path):
                    if limit is not None and df.shape[0] >= limit:
                        break
                    df = pd.concat([df, pd.read_csv(open(os.path.join(path, file), 'r'), **CSV_TO_PD_KWARGS)])
            else:
                df = pd.read_csv(open(path, 'r'), **CSV_TO_PD_KWARGS)
            df = df.iloc[: limit] if limit is not None else df
            return [TaggedDocument(row['text'].split(), [i]) for i, row in df.iterrows()]
        if text is not None:
            return [TaggedDocument(rev.split(), [i]) for i, rev in enumerate(text.split('\n'))]

    def save(self, name):
        save_path = os.path.join("../textVectorizationModels", f"d2v_{name}.model")
        print(f"Saving to {save_path} ...")
        self.vec.save(save_path)

    def train(self, text):
        return self.vec.train(self._as_tagged_doc(text=text))

    def transform(self, text):
        return self.vec.wv[text]

    def get_feature_names(self):
        return self.vec.wv.vocab.keys()


class FT(Vectorizer):
    def __init__(self, path, load=False, limit=None):
        if load:
            self.vec = FastText.load(path)
        else:
            self.vec = FastText()

    def train(self, text):
        return self.vec.train(sentences=text)

    def transform(self, text):
        return self.vec.wv[text]

    def get_feature_names(self):
        return self.vec.wv.vocab.keys()


class ComplexT2v(Vectorizer):

    def __init__(self):
        self._tfidf = T2v()
        self._w2v = W2v()

    def train(self, text):
        self._tfidf.train(text)
        self._w2v.train(text)
        return self.transform(text)

    def transform(self, text):
        pass

    def measure_accuracy(self, X_test, y_test, top_n: int = 5):
        pass

    def get_feature_names(self):
        pass


def test_w2v(path_to_data, load=False, local=True, limit=100, save=True):
    model = W2v(path_to_data, load, local, limit)
    vocab = model.get_feature_names()
    # model.transform(pd.DataFrame(['davinci'], columns=['text']))
    print(model.wv.most_similar_cosmul(positive=['abrasive', 'good']))
    # print(model.wv.most_similar_cosmul(positive=['good'], negative=['bad']))
    # print(model.wv.most_similar(positive=['bad'], negative=['more']))
    if save:
        model.save(f"{os.path.basename(path_to_data).split('.')[0]}_{limit if limit is not None else 0}")


def test_t2v(df, how='tfidf'):
    vectorizer = T2v(how)
    X, y = df.text, df.score
    X = vectorizer.train(X)
    vectorizer.measure_accuracy(X, y, 10)
    # vectorizer.save("t2v_50k.model")


def test_d2v(path_to_data, lim=100, load=False, save=True):
    vec = D2v(path_to_data, load, lim)
    with open('../dev/csv/movie_reviews_20.csv', 'r') as f:
        rand_row = random.choice(f.readlines())
    sep = rand_row.find(',')
    s, r = float(rand_row[: sep]), rand_row[sep:]
    vec.vec.most_similar_cosmul(positive=[r])
    vec.vec.most_similar(positive=[r])
    if save:
        vec.save(f"{os.path.basename(path_to_data).split('.')[0]}_{lim}")


if __name__ == '__main__':
    # ======= sample all data =======
    # df = pd.DataFrame(ReviewGenerator("../csv"))

    # ======= sample balanced data =======
    # df = pd.concat([pd.DataFrame(BalancedReviewGenerator(f"movies_{i}", 1000, True))
    #                 for i in range(1, 11)])
    df = pd.concat([BinaryReviewGenerator("../csv/bin_train_1.csv").as_df(),
                    BinaryReviewGenerator("../csv/bin_test_1.csv").as_df()])

    print(f"balance:\n{df.score.value_counts()}")

    # test_w2v(r"textVectorizationModels\csv_80.model", load=True, save=False)
    # test_d2v("../csv/movie_reviews_1.csv", lim=10000, save=False)
    test_t2v(df)
