import os
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
        data.write_to_file('../csv/{0}_{1}.txt'.format(path.split('.')[0].split('/')[1], i + 1))


def convert_to_csv(src, dst):
    """
    Converts files written in the old output format to csv format
    :param src: source directory
    :param dst: destination directory
    """
    files_data = [FileData(STEP_SIZE, path=os.path.join(src, f), encoding=LARGE_INPUT_FILE_ENCODING) for f
                  in os.listdir(src)]
    for i, data in enumerate(files_data):
        data.write_to_file(os.path.join(dst, 'movies_{0}.txt'.format(i + 1)))


def main():
    # parse_file(LARGE_INPUT_FILE)
    convert_to_csv('../out', '../csv')


if __name__ == '__main__':
    main()
