from FileData import ReviewData


class Parser:

    def __init__(self, path: str, encoding: str):
        self._file = path
        self._encoding = encoding

    def can_parse(self):
        pass

    def next_review(self):
        pass


class LargeFileParser(Parser):

    def __init__(self, path: str, encoding: str):
        Parser.__init__(self, path, encoding)
        self.__line_cursor = 0
        self.__seek_position = 0
        self.__curr_review = 1
        self.__num_reviews = 0

        with open(self._file, 'r', encoding=self._encoding) as f:
            text = f.readlines()

        for line in text:
            if line.startswith('product'):
                self.__num_reviews += 1

    def can_parse(self):
        return self.__curr_review <= self.__num_reviews

    def next_review(self):
        if not self.can_parse():
            return
        with open(self._file, 'r', encoding=self._encoding) as f:
            f.seek(self.__seek_position)
            score = -1
            text = None
            line = f.readline()
            self.__line_cursor += 1
            while line != '\n' and line:
                line = f.readline()
                self.__line_cursor += 1

                if line.startswith('review/score'):
                    score = float(line.split()[1])
                elif line.startswith('review/text'):
                    text = line[len('review/text: '):]

            self.__seek_position = f.tell()
            if score > 0 and text:
                self.__curr_review += 1
                return ReviewData(score, text)

    def get_line_cursor(self):
        return self.__line_cursor

    def get_encoding(self):
        return self._encoding
