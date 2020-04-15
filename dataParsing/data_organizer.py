import os
import re
import time
import pandas as pd
from nltk import pos_tag
from dataParsing.Parser import LargeFileParser
from dataParsing.FileData import *
from ReviewGenerator import ReviewGenerator

LARGE_INPUT_FILE = '../ml-dataset/movies.txt'
LARGE_INPUT_FILE_ENCODING = 'iso-8859-1'
STEP_SIZE = 100000
CSV_TO_PD_KWARGS = {'delimiter': ',', 'quotechar': '"', 'escapechar': '\\'}
PD_TO_CSV_KWARGS = {'quotechar': '"', 'escapechar': '\\', 'quoting': csv.QUOTE_NONNUMERIC, 'index': False}


def parse_file(path, step_size=STEP_SIZE, encoding=LARGE_INPUT_FILE_ENCODING):
    """
    Parses the Amazon movie review data set into FileData objects and then writes each of them to csv
    :param path: path to Amazon data set
    :param step_size: amount of reviews to store in each FileData object
    :param encoding: of the amazon data set
    """
    file_data = []
    with open(path, 'r', encoding=encoding) as f:
        for i, l in enumerate(f):
            pass
    length = i + 1

    remaining = length
    parser = LargeFileParser(path, encoding)
    while remaining > 0:
        file_data.append(FileData(parser=parser, max_reviews=step_size))
        remaining = length - parser.get_line_cursor()

    for i, data in enumerate(file_data):
        data.write_to_file(f"../csv/{path.split('.')[0].split('/')[1]}_{i + 1}.txt")


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


def sort_by_rating(src, dst):
    df = pd.read_csv(open(src, 'r'), delimiter=',', quotechar='"', escapechar='\\')
    df.sort_values('score', ascending=False, inplace=True)
    for s in set(df['score']):
        filename = f"{os.path.basename(src).split('.')[0]}_score{int(s)}.csv"
        df.loc[df.score == s].reset_index(drop=True).to_csv(os.path.join(dst, filename), columns=['text'],
                                                            **PD_TO_CSV_KWARGS)


def is_numeric(s):
    try:
        float(s) if '.' in s else int(s)
        return True
    except ValueError:
        return False


def count_wrapper(func):
    def wrapper(*args, **kwargs):
        ret_val = func(*args, **kwargs)
        if ret_val is False:
            wrapper.filtered += 1
            print(wrapper.filtered)
        return ret_val

    wrapper.filtered = 0
    return wrapper


@count_wrapper
def contains_adj(s):
    adjectives = ['JJ']
    pos = pos_tag(s)
    for w in pos:
        if w[1] in adjectives:
            return True
    return False


def clean_data(dir_path):
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f)) and f.endswith('.csv')]
    for file in files:
        file_path = os.path.join(dir_path, file)
        print(f"cleaning {file}...")
        # remove specific terms
        with open(file_path, 'r') as f:
            s = f.read()
            # s = re.sub(r' *br([^a-zA-z])', r'\g<1>', s)  # br -> ''
            # s = re.sub(r' don ', r' dont ', s)  # don -> dont
            s = re.sub(r' ve ', r' ive ', s)  # ve -> ive
            s = re.sub(r' quot ', r'', s)  # quot -> ''
        with open(file_path, 'w') as f:
            f.write(s)
        # load to DataFrame
        df = pd.read_csv(open(file_path), **CSV_TO_PD_KWARGS)
        # drop nan or empty values
        # df.dropna(inplace=True)
        # drop non numeric scores
        # df = df[df.score.apply(lambda x: is_numeric(str(x)))]
        # drop reviews with no adjectives
        # df = df[df.text.apply(lambda x: contains_adj(x))]

        # df['score'] = df['score'].astype(float)
        df.to_csv(file_path, **PD_TO_CSV_KWARGS)


def main():
    clean_data('../csv')
    for f in os.listdir("../csv"):
        print(f"sorting {f}...")
        sort_by_rating(os.path.join("../csv", f), "../reviewsByStarRating")


if __name__ == '__main__':
    main()
