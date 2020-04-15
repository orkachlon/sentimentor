import os
import random
import pandas as pd


class ReviewGenerator:

    def __init__(self, path, limit=None, shuffle=False):
        self.__path = path
        self.__limit = None if limit is None else max(limit, 0)
        self.__iter_mode = 'row'
        self.__shuffle = shuffle

    def set_lim(self, limit):
        self.__limit = limit

    def iter_scores(self):
        self.__iter_mode = 'score'
        return iter(self)

    def iter_revs(self):
        self.__iter_mode = 'review'
        return iter(self)

    def iter_rows(self):
        self.__iter_mode = 'row'
        return iter(self)

    def __iter__(self):
        if os.path.isdir(self.__path):
            print("Generating reviews from {0}".format(self.__path))
            file_list = os.listdir(self.__path)
            # shuffle files for some variance
            if self.__shuffle:
                random.shuffle(file_list)
            for file in file_list:
                file_path = os.path.join(self.__path, file)
                if os.path.isfile(file_path):
                    df = pd.read_csv(open(file_path, 'r'), delimiter=',', quotechar='"', escapechar='\\', header=0)
                    for j, row in df.iterrows():
                        if j == self.__limit:
                            return
                        yield row if self.__iter_mode == 'row' else row['score'] if self.__iter_mode == 'score' else row['text']
        else:
            print(f"Generating reviews from {self.__path}")
            df = pd.read_csv(open(self.__path, 'r'), delimiter=',', quotechar='"', escapechar='\\', header=0)
            for j, row in df.iterrows():
                if j == self.__limit:
                    return
                yield row if self.__iter_mode == 'row' else row['score'] if self.__iter_mode == 'score' else row['text']


if __name__ == '__main__':
    text_gen = ReviewGenerator("csv/movie_reviews_1.csv")
    i = 0
    for r in text_gen.iter_rows():
        i += 1
    print(f"finished iterating over {i} reviews")
    # d = pd.DataFrame(text_gen)
    # print(d.shape)
    # print(d.head())
