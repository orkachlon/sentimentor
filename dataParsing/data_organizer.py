import os
import pandas as pd
import time
from dataParsing.Parser import LargeFileParser
from dataParsing.FileData import *

LARGE_INPUT_FILE = '../ml-dataset/movies.txt'
LARGE_INPUT_FILE_ENCODING = 'iso-8859-1'
STEP_SIZE = 100000


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
    df.dropna(axis=0, inplace=True, how='any')
    for s in set(df['score']):
        filename = f"{os.path.basename(src).split('.')[0]}_score{int(s)}.csv"
        df.loc[df.score == s].reset_index(drop=True).to_csv(os.path.join(dst, filename), columns=['text'],
                                                            index=False, quotechar='"', escapechar='\\')


def main():
    start = time.time()
    # parse_file(LARGE_INPUT_FILE)
    # convert_to_csv('../out', '../csv')
    files = [f for f in os.listdir('../csv') if os.path.isfile(f"../csv/{f}")]
    for i, f in enumerate(files):
        print(f"{(i / len(files) * 100):.2f}% splitting {f}...")
        sort_by_rating(f'../csv/{f}', "../reviewsByStarRating")
    print(f"time: {(time.time() - start):.2f} s")


if __name__ == '__main__':
    main()
