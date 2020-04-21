import os
import re
import mmap
import contextlib
from abc import ABC, abstractmethod
from dev.preprocessing.FileData import ReviewData


class Parser(ABC):

    def __init__(self, path: str, encoding: str):
        self._file = path
        self._encoding = encoding

    @abstractmethod
    def can_parse(self):
        pass

    @abstractmethod
    def next_review(self):
        pass


class LargeFileParser(Parser):

    def __init__(self, path: str, encoding: str):
        Parser.__init__(self, path, encoding)
        self.__line_cursor = 0
        self.__seek_position = 0
        self.__curr_review = 0
        self.__num_reviews = 0

        if os.path.exists("../ml-dataset/file_info.txt"):
            with open("../ml-dataset/file_info.txt", 'r') as info_file:
                info = info_file.read()
                self.__curr_review = int(re.search(r'curr_review: ([0-9]+)', info).group(1))
                self.__num_reviews = int(re.search(r'num_reviews: ([0-9]+)', info).group(1))
        else:
            with open(self._file, 'r', encoding=self._encoding) as f:
                text = f.readlines()
            for line in text:
                if line.startswith('product'):
                    self.__num_reviews += 1
            with open('../ml-dataset/file_info.txt', 'a') as f:
                f.write(f"num_reviews: {self.__num_reviews}")

        scores = re.compile(br'review/score: ([0-9])\.[0-9]\n')
        texts = re.compile(br'review/text: (.*)\n\n')
        with open(path, 'r', encoding=encoding) as ifile:
            with contextlib.closing(mmap.mmap(ifile.fileno(), 0, access=mmap.ACCESS_READ)) as mp:
                self._scores = [int(m) for m in scores.findall(mp)]
                self._texts = texts.findall(mp)
        assert len(self._scores) == len(self._texts) == self.__num_reviews

    def can_parse(self):
        return self.__curr_review < self.__num_reviews

    def next_review(self):
        if not self.can_parse():
            return
        rev = ReviewData(self._scores[self.__curr_review], self._texts[self.__curr_review])
        self.__curr_review += 1
        return rev
        # with open(self._file, 'r', encoding=self._encoding) as f:
        #     f.seek(self.__seek_position)
        #     score = -1
        #     text = None
        #     line = f.readline()
        #     self.__line_cursor += 1
        #     while line != '\n' and line:
        #         line = f.readline()
        #         self.__line_cursor += 1
        #
        #         if line.startswith('review/score'):
        #             score = float(line.split()[1])
        #         elif line.startswith('review/text'):
        #             text = line[len('review/text: '):]
        #
        #     self.__seek_position = f.tell()
        #     if score > 0 and text:
        #         self.__curr_review += 1
        #         return ReviewData(score, text)

    def __iter__(self):
        tmp = self.__curr_review
        for i in range(tmp, self.__num_reviews):
            self.__curr_review += 1
            yield self._scores[i], self._texts[i]

    def __getitem__(self, index):
        return self._scores[index], self._texts[index]

    def __len__(self):
        return self.__num_reviews

    def get_line_cursor(self):
        return self.__line_cursor

    def get_encoding(self):
        return self._encoding

    def get_curr_review_n(self):
        return self.__curr_review
