from dataParsing.Parser import LargeFileParser
from dataParsing.FileData import *


LARGE_INPUT_FILE = '../ml-dataset/movies.txt'
LARGE_INPUT_FILE_ENCODING = 'iso-8859-1'
STEP_SIZE = 100000


def parse_file(path):
    remaining = 0
    file_data = []
    with open(path, 'r', encoding=LARGE_INPUT_FILE_ENCODING) as f:
        for i, l in enumerate(f):
            pass
    length = i + 1

    remaining = length
    parser = LargeFileParser(path, LARGE_INPUT_FILE_ENCODING)
    while remaining > 0:
        file_data.append(FileData(parser=parser, step_size=STEP_SIZE))
        remaining = length - parser.get_line_cursor()

    for i, data in enumerate(file_data):
        data.write_to_file('../out/{0}_{1}.txt'.format(path.split('.')[0].split('/')[1], i + 1))


def main():
    parse_file(LARGE_INPUT_FILE)


if __name__ == '__main__':
    main()
