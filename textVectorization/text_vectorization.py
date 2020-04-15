import os
# import sys
import random
import pandas as pd
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
from gensim.models.fasttext import FastText
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer

from ReviewGenerator import ReviewGenerator
from dataParsing.data_organizer import PD_TO_CSV_KWARGS, CSV_TO_PD_KWARGS

WORD_SPACE_DIM = 300


class Vectorizer:
    def train(self, text):
        pass

    def transform(self, text):
        pass

    def get_feature_names(self):
        pass


class W2v(Vectorizer):
    def __init__(self, path: str, load=False, local=True, limit: int = None):
        if load:
            self.vec = Word2Vec.load(path) if local else api.load(path)
        elif path:
            sentences = ReviewGenerator(path, limit=limit) if local else api.load(path)
            self.vec = Word2Vec(sentences=sentences, size=WORD_SPACE_DIM, workers=4)
        else:
            self.vec = Word2Vec(size=WORD_SPACE_DIM, workers=4)
        self.wv = self.vec.wv

    def train(self, text):
        self.vec.train(text)
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
        return self.vec.wv[text]

    def get_feature_names(self):
        return self.vec.wv.vocab


class T2v(Vectorizer):

    def __init__(self, how='tfidf'):
        if how == 'tfidf':
            self.vec = TfidfVectorizer(**{'sublinear_tf': True, 'norm': 'l2', 'encoding': 'latin-1',
                                       'ngram_range': (1, 2), 'stop_words': 'english', 'min_df': .03, 'max_df': .2})
        elif how == 'count':
            self.vec = CountVectorizer(**{'encoding': 'latin-1', 'ngram_range': (1, 2),
                                          'stop_words': 'english', 'min_df': .07, 'max_df': .2})
        else:
            print(f"Method '{how}' is unrecognized, no vectorizer was made")
            self.vec = None

    def train(self, text):
        if self.vec is not None:
            return self.vec.fit_transform(text).toarray()

    def transform(self, text):
        if self.vec is not None:
            return self.vec.transform(text).toarray()

    def get_feature_names(self):
        return self.vec.get_feature_names()


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


def test_w2v(path_to_data, load=False, local=True, limit=100, save=True):
    model = W2v(path_to_data, load, local, limit)
    print(len(model.get_feature_names()))
    if save:
        model.save(f"{os.path.basename(path_to_data).split('.')[0]}_{limit if limit is not None else 0}")


def test_t2v(path_to_data, lim=100, how='tfidf'):
    text_gen = ReviewGenerator(path_to_data, lim)
    data = pd.DataFrame(text_gen)
    train, test = data[: int(data.shape[0] * .75)], data[int(data.shape[0] * .75):]
    vectorizer = T2v(how)
    train_vecs = vectorizer.train((row['text'] for _, row in train.iterrows()))
    test_vecs = vectorizer.transform((row['text'] for _, row in test.iterrows()))
    print(train_vecs.shape, test_vecs.shape)


def test_d2v(path_to_data, lim=100, load=False, save=True):
    vec = D2v(path_to_data, load, lim)
    with open('../csv/movie_reviews_20.csv', 'r') as f:
        rand_row = random.choice(f.readlines())
    sep = rand_row.find(',')
    s, r = float(rand_row[: sep]), rand_row[sep:]
    vec.vec.most_similar_cosmul(positive=[r])
    vec.vec.most_similar(positive=[r])
    if save:
        vec.save(f"{os.path.basename(path_to_data).split('.')[0]}_{lim}")


if __name__ == '__main__':
    # if not 3 <= len(sys.argv) <= 4:
    #     print('Usage: python model <path-to-data-or-model> -n <optional-file-limit>')
    #     exit(-1)
    # test_w2v(r"../textVectorizationModels/text8.model", load=True, save=False)
    # test_d2v("../csv/movie_reviews_1.csv", lim=10000, save=False)
    test_t2v(r'../csv/movie_reviews_1.csv', lim=1000)
