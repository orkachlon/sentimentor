import os
import random
import pandas as pd


class ReviewGenerator:

    def __init__(self, path, limit=None, shuffle=True):
        self._path = path
        self._limit = None if limit is None else max(limit, 0)
        self._iter_values = 'row'
        self._shuffle = shuffle

    def set_lim(self, limit):
        self._limit = limit

    def iter_scores(self):
        self._iter_values = 'y'
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
                        yield row if values == 'row' else row['y'] if values == 'y' else row['text']
        else:
            df = pd.read_csv(open(self._path, 'r'), delimiter=',', quotechar='"', escapechar='\\', header=0)
            if self._shuffle:
                df = df.sample(frac=1).reset_index(drop=True)
            for j, row in df.iterrows():
                if j == self._limit:
                    return
                yield row if values == 'row' else row['y'] if values == 'y' else row['text']


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
            yield row if values == 'row' else row['y'] if values == 'y' else row['text']

    def balance_data(self):
        """
        Set the limit to the minimum amount of data from a single y
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
                         delimiter=',', quotechar='"', header=0, names=['y', 'text'], encoding='cp1252')
        if self._shuffle:
            df = df.sample(frac=1).reset_index(drop=True)
        for i, row in df.iterrows():
            if i == self._limit:
                break
            yield row if values == 'row' else row['text'] if values == 'text' else row['y']

    def as_df(self):
        df = pd.read_csv(open(self._path, 'r'),
                         delimiter=',', quotechar='"', header=0, names=['y', 'text'], encoding='cp1252')
        if self._limit:
            df = pd.concat([df.loc[df.score == -1].iloc[: int(self._limit / 2)],
                            df.loc[df.score == 1].iloc[: int(self._limit / 2)]])
        return df if not self._shuffle else df.sample(frac=1).reset_index(drop=True)


def main() -> None:
    """
    Performs a sanity check for each generator
    """
    # test ReviewGenerator
    text_gen = ReviewGenerator("../assets/csv/movie_reviews_1.csv", limit=10)
    i = 0
    print("ReviewGenerator:")
    for row in text_gen:
        print(f"#{i + 1}")
        print(row)
        i += 1
    # test BalancedReviewGenerator
    balanced_gen = BalancedReviewGenerator('movies_1', 10)
    print("BalancedReviewGenerator:")
    i = 0
    for row in balanced_gen:
        print(f"#{i + 1}")
        print(row)
        i += 1
    # test BinaryReviewGenerator
    binary_gen = BinaryReviewGenerator("../assets/raw/bin_train.csv", 10)
    print("BinaryReviewGenerator:")
    i = 0
    for row in binary_gen:
        print(f"#{i + 1}")
        print(row)
        i += 1


if __name__ == '__main__':
    main()
