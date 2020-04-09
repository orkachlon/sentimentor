class FileData:
    """
    A container for ReviewData objects. Can read reviews from file or use parser to parse the original file
    """
    def __init__(self, step_size: int, parser=None, path=None):
        self.__reviews = []
        self.__parser = parser
        if path is not None:
            self.__read_from_file(path)
        elif parser is not None:
            i = 0
            while parser.can_parse() and i < step_size:
                self.__reviews.append(parser.next_review())
                i += 1

    def __repr__(self):
        return str(self.__reviews)

    def __read_from_file(self, path):
        with open(path, 'r') as f:
            for line in f:
                if line.startswith('score'):
                    score = float(line.split()[1])
                elif line.startswith('text'):
                    text = line[len('text:  '):]
                    self.__reviews.append(ReviewData(score, text))

    def write_to_file(self, filename):
        if self.__parser is not None:
            with open(filename, 'w', encoding=self.__parser.get_encoding()) as f:
                f.writelines([str(review) for review in self.__reviews])
        else:
            with open(filename, 'w') as f:
                f.writelines([str(review) for review in self.__reviews])


class ReviewData:
    """
    Holds information of a review
    """

    def __init__(self, score: float, text: str):
        self.__score = score
        self.__text = text

    def __repr__(self):
        return 'score: {0}\ntext:  {1}\n'.format(self.__score, self.__text)

    def __str__(self):
        return 'score: {0}\ntext:  {1}\n'.format(self.__score, self.__text)

    def get_score(self):
        return self.__score

    def get_text(self):
        return self.__text
