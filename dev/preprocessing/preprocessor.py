import os
import re
import nltk
import pandas as pd
from string import punctuation
from nltk import pos_tag
from nltk.corpus import stopwords
from spellchecker import SpellChecker

from preprocessing.Parser import FileParser
from preprocessing.FileData import *

LARGE_INPUT_FILE = '../ml-dataset/movies.txt'
LARGE_INPUT_FILE_ENCODING = 'iso-8859-1'
STEP_SIZE = 100000
CSV_TO_PD_KWARGS = {'delimiter': ',', 'quotechar': '"', 'escapechar': '\\', 'header': 0}
PD_TO_CSV_KWARGS = {'quotechar': '"', 'escapechar': '\\', 'quoting': csv.QUOTE_NONNUMERIC, 'index': False}
# regex to add quotes around text in pos/neg review files
# ADD_QUOTES = re.compile(rb'^(([^"])(.*)([^"])),(pos|neg)$', re.MULTILINE)


def parse_file(path: str, step_size: int = STEP_SIZE, encoding: str = LARGE_INPUT_FILE_ENCODING, cleaner=None):
    """
    Parses the Amazon movie review data set into FileData objects and then writes each of them to csv
    :param path: path to Amazon data set
    :param step_size: amount of reviews to store in each FileData object
    :param encoding: of the amazon data set
    :param cleaner: a text cleaning function to clean the text before writing it back
    """
    _, file_n = get_file_info(path, os.path.basename(path), encoding)

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


def get_file_info(path: str, filename: str, encoding: str = None):
    info_pat = re.compile(r'file_name: (' + filename +
                          r')\nnum_lines: (?P<num_lines>[0-9]+)'
                          r'\nfile_n: (?P<file_n>[0-9]+)\n')
    if os.path.exists("../ml-dataset/file_info.txt"):
        with open("../ml-dataset/file_info.txt", 'r') as info_file:
            info = info_file.read()
        m = info_pat.search(info)
        if m is not None:
            return int(m.group('num_lines')), int(m.group('file_n'))
    with open(path, 'r', encoding=encoding) as f:
        for i, l in enumerate(f):
            pass
    num_lines = i + 1
    file_n = 1
    return num_lines, file_n


def dump_info_file(parser, file_n):
    with open("../ml-dataset/file_info.txt", 'r+') as info_file:
        info = info_file.read()
        info_file.seek(0)
        info = re.sub(r'curr_review: [0-9]+\n', f'curr_review: {parser.get_curr_review_n() - 1}\n', info)
        info = re.sub(r'file_n: [0-9]+\n', f'file_n: {file_n}\n', info)
        info_file.write(info)


def convert_to_csv(src, dst):
    """
    Converts files written in the old output format to csv format
    :param src: source directory
    :param dst: destination directory
    """
    files = [f for f in os.listdir(src) if os.path.isfile(os.path.join(src, f)) and f.endswith('.txt')]
    for i, f in enumerate(files):
        print(f"{((i / len(files)) * 100):.2f}%")
        data = FileData(STEP_SIZE, path=os.path.join(src, f), encoding=LARGE_INPUT_FILE_ENCODING)
        data.write_to_file(os.path.join(dst, f"movie_reviews_{i + 1}.csv"))


def sort_by_rating(src_dir, dst_dir):
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

    def text_cleaner(s: bytes):
        s = re.sub(r'\\p{L1}}', r'', s.decode('cp1252', 'ignore')).encode('cp1252')
        if alpha_numeric:
            s = non_alpha_numeric.sub(rb'', s)
        if lower:
            s = s.lower()
        if html:
            s = html_cleaner.sub(b'', s)
        if punct:
            # method 1:
            # s = re.sub(rb"['`\".:;?!@#$%^&*(){}\-=+_~/<>,\\|\[\]]", rb'', s)
            # method 2:
            s = punct_cleaner.sub(br' ', s)
            # method 3:
            # words = nltk.word_tokenize(s.decode(LARGE_INPUT_FILE_ENCODING))
            # s = ' '.join([w for w in words if w not in punctuation])
            # method 4:
            # s = s.decode(LARGE_INPUT_FILE_ENCODING, 'ignore')
            # s = s.translate(str.maketrans('', '', punctuation))
        if num:
            s = num_cleaner.sub(rb'', s)
        if stop:
            # method 1: DOES NOT WORK FOR BI-GRAMS
            # ================
            # try:
            #     words = nltk.word_tokenize(s.decode('cp1252', 'ignore'))
            # except AttributeError:
            #     words = nltk.word_tokenize(s)
            # s = ' '.join([w for w in words if w not in stopwords_set])

            # method 2:
            # ================
            s = b' ' + s + b' '
            s = stopwrds_cleaner.sub(br'', s)
            s = re.sub(rb' +', b' ', s)
            s = s.strip()
        if spell:
            try:
                words = nltk.word_tokenize(s.decode('cp1252', 'ignore'))
            except AttributeError:
                words = nltk.word_tokenize(s)
            splchkr = SpellChecker()
            s = ' '.join(list(map(splchkr.correction, words)))
        try:
            s = s.decode('cp1252', 'ignore')
        except AttributeError:
            pass
        return s
    return text_cleaner


def main():
    parse_file('../ml-dataset/bin_test.csv',
               cleaner=get_text_cleaner(stop=False, punct=False, num=False, spell=False, lower=False),
               encoding='utf-8')
    # sort_by_rating("../csv", "../reviewsByStarRating")


if __name__ == '__main__':
    main()
