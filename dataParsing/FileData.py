from gensim import utils
import csv


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
        self.__reviews = []
        self.__parser = parser
        if max_reviews <= 0:
            return
        if path is not None and encoding is not None:
            self.__encoding = encoding
            self.__read_from_file(path, max_reviews)
        elif parser is not None:
            i = 0
            while parser.can_parse() and i < max_reviews:
                self.__reviews.append(parser.next_review())
                i += 1

    def __repr__(self):
        """
        :return: Representation of contained reviews
        """
        return str(self.__reviews)

    def __read_from_file(self, path, limit):
        """
        Reads the reviews from a csv file
        :param path: path to reviews file
        ":param limit: maximum number of reviews to read
        """
        self.__reviews = [ReviewData(row['score'], row['text']) for row in
                          csv.DictReader(path, quotechar='"', escapechar='\\', sep=',')[1: limit]]
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
                f.write('score,text\n')
                f.writelines([str(review) for review in self.__reviews])
        if format == 'ls':  # LineSentence format
            with open(filename, 'w', encoding=encoding) as f:
                f.writelines([review.get_text() for review in self.__reviews])


class ReviewData:
    """
    Holds information of a review
    """

    def __init__(self, score: float, text: str):
        """
        Initializes the review with a score and its text
        :param score: 1-5 (float)
        :param text: review (str)
        """
        self.__score = score
        self.__text = ' '.join(utils.simple_preprocess(text)).replace('"', r'\"') + '\n'

    def __repr__(self):
        """
        :return: Readable representation of this review
        """
        return 'score: {0}\ntext: "{1}"\n'.format(self.__score, self.__text)

    def __str__(self):
        """
        :return: String representation of this review. Format is CSV.
        """
        return '{0},"{1}"\n'.format(self.__score, self.__text)

    def get_score(self):
        """
        :return: Review score
        """
        return self.__score

    def get_text(self):
        """
        :return: Review text
        """
        return self.__text
