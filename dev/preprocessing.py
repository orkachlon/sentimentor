import os
import re
import nltk
import pandas as pd
import csv

from typing import Tuple, Callable, Union
from string import punctuation
from nltk.corpus import stopwords
from spellchecker import SpellChecker

from Parser import FileParser

LARGE_INPUT_FILE = '../ml-dataset/movies.txt'
LARGE_INPUT_FILE_ENCODING = 'iso-8859-1'
STEP_SIZE = 100000
CSV_TO_PD_KWARGS = {'delimiter': ',', 'quotechar': '"', 'escapechar': '\\', 'header': 0}
PD_TO_CSV_KWARGS = {'quotechar': '"', 'escapechar': '\\', 'quoting': csv.QUOTE_NONNUMERIC, 'index': False}


def parse_file(path: str,
               step_size: int = STEP_SIZE,
               encoding: str = LARGE_INPUT_FILE_ENCODING,
               cleaner: Callable[[Union[str, bytes]], str] = None):
    """
    Parses the Amazon movie review data set into FileData objects and then writes each of them to csv
    :param path: path to Amazon data set
    :param step_size: amount of reviews to store in each FileData object
    :param encoding: of the amazon data set
    :param cleaner: a text cleaning function to clean the text before writing it back
    """
    _, file_n = get_file_info(path, os.path.basename(path))

    print("Reading input file...")
    parser = FileParser(path, encoding)
    remaining = len(parser)
    base_name = os.path.basename(path).split('.')[0]

    prev = 0
    while remaining > 0:
        increment = step_size if remaining > step_size else remaining
        print(f"{(len(parser) - remaining + increment) / len(parser) * 100:.0f}%")
        with open(f"../csv/{base_name}_{file_n}.csv", 'w') as f:
            f.write('"score","text"\n')
            if cleaner is not None:
                f.write('\n'.join(['%d,"%s"' % (score, cleaner(rev))
                                   for score, rev in zip(*parser[prev: prev + increment])]))
            else:
                f.write('\n'.join(['%d,"%s"' % (score, rev)
                                   for score, rev in zip(*parser[prev: prev + increment])]))
        prev += increment
        remaining -= increment
        file_n += 1


def get_file_info(path: str, filename: str) -> Tuple[int, int]:
    """
    :param path: path to the dataset in case no info is available for it in the info file
    :param filename: name of the dataset file
    :return: tuple: (amount of lines in file, the next file number to save to)
    """
    info_pat = re.compile(r'file_name: (' + filename +
                          r')\nnum_lines: (?P<num_lines>[0-9]+)'
                          r'\nfile_n: (?P<file_n>[0-9]+)\n')
    if os.path.exists("../assets/raw/file_info.txt"):
        with open("../assets/raw/file_info.txt", 'r') as info_file:
            info = info_file.read()
        m = info_pat.search(info)
        if m is not None:
            return int(m.group('num_lines')), int(m.group('file_n'))
    with open(path, 'r') as f:
        for i, l in enumerate(f):
            pass
    num_lines = i + 1
    file_n = 1
    return num_lines, file_n


def sort_by_rating(src_dir: str, dst_dir: str) -> None:
    """
    Used to split the 8M Amazon review dataset into files containing reviews only from one score
    :param src_dir: directory to the dataset as csv files
    :param dst_dir: where to save the split files
    """
    for src in os.listdir(src_dir):
        print(f"sorting {src}...")
        src_path = os.path.join(src_dir, src)
        df = pd.read_csv(open(src_path, 'r'), **CSV_TO_PD_KWARGS)
        for s in set(df['score']):
            filename = f"{src.split('.')[0]}_score{int(s)}.csv"
            df.loc[df.score == s].reset_index(drop=True).to_csv(os.path.join(dst_dir, filename),
                                                                columns=['score', 'text'],
                                                                **PD_TO_CSV_KWARGS)


def get_text_cleaner(stop=True, punct=True, num=True, html=True, spell=True, alpha_numeric=False, lower=True):
    """
    Defines and returns a cleaner function according to given params
    :param stop: remove stopwords
    :param punct: remove punctuation
    :param num: remove numbers
    :param html: remove html tags
    :param spell: correct spelling. Note: this takes a lot of time - never fully tried it.
    :param alpha_numeric: remove any non alpha numeric chars
    :param lower: convert all to lower case
    :return: a function that does what the parameters specify
    """
    # define all patterns here to do it only once
    punct_cleaner = re.compile(br"[" + re.escape(punctuation).encode('cp1252', 'ignore') + br"]")
    html_cleaner = re.compile(rb'<.*?>|&([a-z0-9]+|#[0-9]{1,6}|#x[0-9a-f]{1,6});', re.MULTILINE)
    num_cleaner = re.compile(rb'([0-9](\.[0-9]*)?)+')
    non_alpha_numeric = re.compile(rb'[^0-9a-zA-Z\s]')
    stopwords_set = stopwords.words('english')
    stopwords_set += ['special effects', 'dvd', 'ive seen', 'quot', 'amp']
    stopwords_set += [w.replace(r"'", "") for w in stopwords_set if "'" in w]
    stopwords_set = set(stopwords_set)
    stopwrds_cleaner = re.compile(rb'(?<= )(' +
                                  '|'.join(stopwords_set).encode('cp1252', 'ignore') +
                                  br')(?= )')
    splchkr = SpellChecker()

    def text_cleaner(s: bytes) -> str:
        """
        a text cleaning function
        :param s: text to clean
        :return: cleaned text
        """
        s = re.sub(r'\\p{L1}}', r'', s.decode('cp1252', 'ignore')).encode('cp1252')
        if alpha_numeric:
            s = non_alpha_numeric.sub(rb'', s)
        if lower:
            s = s.lower()
        if html:
            s = html_cleaner.sub(b'', s)
        if punct:
            s = punct_cleaner.sub(br' ', s)
        if num:
            s = num_cleaner.sub(rb'', s)
        if stop:
            s = b' ' + s + b' '
            s = stopwrds_cleaner.sub(br'', s)
            s = re.sub(rb' +', b' ', s)
            s = s.strip()
        if spell:
            try:
                words = nltk.word_tokenize(s.decode('cp1252', 'ignore'))
            except AttributeError:
                # noinspection PyTypeChecker
                words = nltk.word_tokenize(s)
            s = ' '.join(list(map(splchkr.correction, words)))
        try:
            s = s.decode('cp1252', 'ignore')
        except AttributeError:
            pass
        return s
    return text_cleaner


def main(path_to_file: str,
         encoding='utf-8',
         **cleaner_kwargs) -> None:
    """
    Cleans specified file
    :param path_to_file: path to file to be cleaned
    :param encoding: encoding of the file
    :param cleaner_kwargs: cleaner parameters
    """
    parse_file(path_to_file,
               cleaner=get_text_cleaner(**cleaner_kwargs),
               encoding=encoding)


if __name__ == '__main__':
    main('../assets/raw/bin_test.csv', stop=False, punct=False, num=False, spell=False, lower=False)
