

class FileData:
    def __init__(self, parser, step_size: int):
        self.__reviews = []
        self.__parser = parser
        i = 0
        while parser.can_parse() and i < step_size:
            self.__reviews.append(parser.next_review())
            i += 1

    def __repr__(self):
        return str(self.__reviews)

    def write_to_file(self, filename):
        with open(filename, 'w', encoding=self.__parser.get_encoding()) as f:
            f.writelines([str(review) for review in self.__reviews])


class ReviewData:
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
