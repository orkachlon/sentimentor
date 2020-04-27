import os
import random
import pandas as pd
from preprocessing.preprocessor import get_text_cleaner


class ReviewGenerator:

    def __init__(self, path, limit=None, shuffle=True):
        self._path = path
        self._limit = None if limit is None else max(limit, 0)
        self._iter_values = 'row'
        self._shuffle = shuffle

    def set_lim(self, limit):
        self._limit = limit

    def iter_scores(self):
        self._iter_values = 'score'
        return iter(self)

    def iter_revs(self):
        self._iter_values = 'review'
        return iter(self)

    def iter_rows(self):
        self._iter_values = 'row'
        return iter(self)

    def __iter__(self):
        values = self._iter_values
        if os.path.isdir(self._path):
            file_list = os.listdir(self._path)
            # shuffle files for more variance
            if self._shuffle:
                random.shuffle(file_list)
            for file in file_list:
                file_path = os.path.join(self._path, file)
                if os.path.isfile(file_path):
                    df = pd.read_csv(open(file_path, 'r'), delimiter=',', quotechar='"', escapechar='\\', header=0)
                    if self._shuffle:
                        df = df.sample(frac=1).reset_index(drop=True)
                    for j, row in df.iterrows():
                        if j == self._limit:
                            return
                        yield row if values == 'row' else row['score'] if values == 'score' else row['text']
        else:
            df = pd.read_csv(open(self._path, 'r'), delimiter=',', quotechar='"', escapechar='\\', header=0)
            if self._shuffle:
                df = df.sample(frac=1).reset_index(drop=True)
            for j, row in df.iterrows():
                if j == self._limit:
                    return
                yield row if values == 'row' else row['score'] if values == 'score' else row['text']


class BalancedReviewGenerator(ReviewGenerator):
    def __init__(self, file_name: str, lim=None, shuffle=True):
        super(BalancedReviewGenerator, self).__init__(f"reviewsByStarRating/{file_name}", lim, shuffle)
        base_path = os.path.join(os.path.dirname(__file__), "reviewsByStarRating")
        self._gens = [ReviewGenerator(os.path.join(base_path, f"{file_name}_score{s}.csv"), limit=int(lim / 5))
                      for s in range(1, 6)]

    def __iter__(self):
        values = self._iter_values
        df = self.as_df()
        if self._shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        for j, row in df.iterrows():
            if j == self._limit:
                return
            yield row if values == 'row' else row['score'] if values == 'score' else row['text']

    def balance_data(self):
        """
        Set the limit to the minimum amount of data from a single score
        """
        df = self.as_df()
        for gen in self._gens:
            gen.set_lim(min(df.score.value_counts()))

    def as_df(self):
        df = pd.concat(pd.DataFrame(gen) for gen in self._gens)
        if self._limit:
            df = pd.concat([df.loc[df.score == s].iloc[: int(self._limit / len(set(df.score)))]
                            for s in set(df.score)])
        return df if not self._shuffle else df.sample(frac=1).reset_index(drop=True)


class BinaryReviewGenerator(ReviewGenerator):
    def __init__(self, path: str, lim=None, shuffle=True):
        super(BinaryReviewGenerator, self).__init__(path, lim, shuffle)

    def __iter__(self):
        values = self._iter_values
        df = pd.read_csv(open(self._path, 'r'),
                         delimiter=',', quotechar='"', header=0, names=['score', 'text'], encoding='cp1252')
        if self._shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        for i, row in df.iterrows():
            if i == self._limit:
                break
            yield row if values == 'row' else row['text'] if values == 'text' else row['score']

    def as_df(self):
        df = pd.read_csv(open(self._path, 'r'),
                         delimiter=',', quotechar='"', header=0, names=['score', 'text'], encoding='cp1252')
        if self._limit:
            df = pd.concat([df.loc[df.score == -1].iloc[: int(self._limit / 2)],
                            df.loc[df.score == 1].iloc[: int(self._limit / 2)]])
        return df if not self._shuffle else df.sample(frac=1).reset_index(drop=True)


def clean_review_generator(generator: ReviewGenerator, mode='row', **cleaner_kwargs):
    cleaner = get_text_cleaner(**cleaner_kwargs)
    for row in generator:
        yield (row['score'], cleaner(row['text'])) if mode == 'row' else cleaner(row['text'])


if __name__ == '__main__':
    binary_gen = BinaryReviewGenerator("ml-dataset/bin_train.csv", 10, False)

    # balanced_gen = BalancedReviewGenerator('movies_1', 300)
    i = 0
    for row in binary_gen:
        print(f"#{i + 1}")
        print(row)
        i += 1
    # text_gen = ReviewGenerator("csv/movie_reviews_1.csv")
    # i = 0
    # for r in text_gen.iter_rows():
    #     i += 1
    # print(f"finished iterating over {i} reviews")
    # d = pd.DataFrame(text_gen)
    # print(d.shape)
    # print(d.head())
