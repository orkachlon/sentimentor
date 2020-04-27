import os
import re
import mmap
import contextlib
from abc import ABC, abstractmethod
from preprocessing.FileData import ReviewData


class Parser(ABC):

    def __init__(self, path: str, encoding: str):
        self._path = path
        self._encoding = encoding

    @abstractmethod
    def can_parse(self):
        pass

    @abstractmethod
    def next_review(self):
        pass


class FileParser(Parser):

    def __init__(self, path: str, encoding: str):
        Parser.__init__(self, path, encoding)
        self.__line_cursor = 0
        self.__seek_position = 0
        self.__curr_review = 0
        self.__num_reviews = 0

        self._load_progress()
        self._read_file()
        assert len(self._scores) == len(self._texts) and len(self._texts) == self.__num_reviews

    def _load_progress(self):
        file_name = os.path.basename(self._path)
        if os.path.exists("../ml-dataset/parser_progress.txt"):
            prog_pat = re.compile(r'file_name: (' + file_name +
                                  r')\nnum_reviews: (?P<num_reviews>[0-9]+)'
                                  r'\ncurr_review: (?P<curr_review>[0-9]+)')
            with open("../ml-dataset/parser_progress.txt", 'r') as info_file:
                info = info_file.read()
            m = prog_pat.search(info)
            if m is not None:
                self.__num_reviews = int(m.group('num_reviews'))
                self.__curr_review = int(m.group('curr_review'))
                return

        with open(self._path, 'r', encoding=self._encoding) as f:
            lines = f.readlines()
        if file_name in ['bin_train.csv', 'bin_test.csv']:
            self.__num_reviews = len([line for line in lines if len(line) > 1])
        elif file_name == 'movies.txt.gz':
            for line in lines:
                if line.startswith('product'):
                    self.__num_reviews += 1

        with open('../ml-dataset/parser_progress.txt', 'a') as f:
            f.write(f"\nfile_name: {file_name}\n")
            f.write(f"num_reviews: {self.__num_reviews}\n")
            f.write("curr_review: 0\n")

    def _read_file(self):
        file_name = os.path.basename(self._path)
        if file_name == 'movies.txt':
            scores = re.compile(br'review/score: ([0-9])\.[0-9]\n')
            texts = re.compile(br'review/text: (.*)\n\n')
            with open(self._path, 'r', encoding=self._encoding) as ifile:
                with contextlib.closing(mmap.mmap(ifile.fileno(), 0, access=mmap.ACCESS_READ)) as mp:
                    self._scores = [int(m) for m in scores.findall(mp)]
                    self._texts = texts.findall(mp)
        elif file_name in ['bin_train.csv', 'bin_test.csv']:
            # scores = re.compile(br',(pos|neg)$', re.MULTILINE)
            # texts = re.compile(br'^"(([^"]|"")(.*)([^"]|""))",', re.MULTILINE)
            reviews = re.compile(br'"(([^"]|"")(.*)([^"]|""))",(pos|neg)', re.MULTILINE)
            with open(self._path, 'rb') as f:
                def numerize(s: bytes) -> int:
                    if s == b'neg':
                        return -1
                    if s == b'pos':
                        return 1
                    return 0
                s = f.read()
                matches = reviews.findall(s)
                self._scores = [numerize(m[4]) for m in matches]
                self._texts = [m[0] for m in matches]

    def can_parse(self):
        return self.__curr_review < self.__num_reviews

    def next_review(self):
        if not self.can_parse():
            return
        rev = ReviewData(self._scores[self.__curr_review], self._texts[self.__curr_review])
        self.__curr_review += 1
        return rev

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
