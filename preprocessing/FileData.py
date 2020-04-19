import csv
from collections import deque


class FileData:
    """
    A container for ReviewData objects. Can read reviews from file or use parser to parse the original file
    """

    def __init__(self, max_reviews=0, parser=None, path=None, encoding='utf-8'):
        """
        Initializes the review container from given source
        :param max_reviews: maximum number of reviews to extract from source
        :param parser:      (optional) a Parser to use for review extraction
        :param path:        (optional) path to a reviews file previously written by this Class
        :param encoding:    (optional) if a path is given then its encoding should be specified.
                            Default is 'utf-8'.
        """
        self.__reviews = deque()
        self.__parser = parser
        if max_reviews <= 0:
            return
        if path is not None and encoding is not None:
            self.__encoding = encoding
            self.__from_csv(path, max_reviews)
        elif parser is not None:
            self.__reviews = [rev for i, rev in enumerate(parser) if i < max_reviews]
            # i = 0
            # while parser.can_parse() and i < max_reviews:
            #     self.__reviews.append(parser.next_review())
            #     i += 1

    def __repr__(self):
        """
        :return: Representation of contained reviews
        """
        return str(self.__reviews)

    def __iter__(self):
        return iter(self.__reviews)

    def __from_csv(self, path, limit, encoding='utf-8'):
        """
        Reads the reviews from a csv file
        :param path: path to reviews file
        ":param limit: maximum number of reviews to read
        """
        with open(path) as f:
            reader = csv.DictReader(f, quotechar='"', escapechar='\\', delimiter=',', skipinitialspace=True)
            for i, row in enumerate(reader):
                if i >= limit:
                    break
                self.__reviews.append(ReviewData(row['score'], row['text']))
        # DEPRECATED
        # with open(path, 'r', encoding=self.__encoding) as f:
        #     for line in f:
        #         if len(self.__reviews) == limit:
        #             break
        #         if line.startswith('score'):
        #             score = float(line.split()[1])
        #         elif line.startswith('text'):
        #             text = line[len('text:  '):]
        #             self.__reviews.append(ReviewData(score, text))

    def write_to_file(self, filename, format='csv'):
        """
        Writes the reviews to a file
        :param filename: path to save the reviews to.
        :param format: 'csv' for CSV or 'ls' for LineSentence. LineSentence doesn't include the score
        """
        encoding = self.__parser.get_encoding() if self.__parser is not None else self.__encoding
        if format == 'csv':  # CSV format
            with open(filename, 'w', encoding=encoding) as f:
                f.write('"score","text"\n')
                f.writelines([f"{str(review)}\n" for review in self.__reviews])
        if format == 'ls':  # LineSentence format
            with open(filename, 'w', encoding=encoding) as f:
                f.writelines([review.text for review in self.__reviews])

    def clean_text(self, cleaner):
        for i in range(len(self.__reviews)):
            self.__reviews[i].text = cleaner(self.__reviews[i].text)


class ReviewData:
    """
    Holds information of a review
    """

    def __init__(self, score: float, text):
        """
        Initializes the review with a score and its text
        :param score: 1-5 (float/int)
        :param text: review (str/bytes)
        """
        self.score = score
        self.text = text

    def __repr__(self):
        """
        :return: Readable representation of this review
        """
        return f'score: {self.score}\ntext: "{self.text}"\n'

    def __str__(self):
        """
        :return: String representation of this review. Format is CSV.
        """
        return f'{self.score},"{self.text.decode("utf-8", "ignore")}"'
