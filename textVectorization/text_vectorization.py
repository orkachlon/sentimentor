import os
# import sys
import gensim
import random
from gensim import utils
from gensim.models import Word2Vec
import gensim.downloader as api

from dataParsing.data_organizer import LARGE_INPUT_FILE_ENCODING

WORD_SPACE_DIM = 300


class TextGenerator:

    def __init__(self, path, limit=0):
        self.__path = path
        self.__limit = max(limit, 0)
        self.__times_iterated = 0

    def __iter__(self):
        self.__times_iterated += 1
        print(f"Starting iteration no. {self.__times_iterated} over data set")
        if os.path.isdir(self.__path):
            print("Folder path given. Generating reviews from {0}".format(self.__path))
            amount = min(len(os.listdir(self.__path)), self.__limit)
            file_list = os.listdir(self.__path)
            random.shuffle(file_list)
            for j, file in enumerate(file_list):
                if j == self.__limit:
                    break
                file_path = os.path.join(self.__path, file)
                if os.path.isfile(file_path):
                    print(f"{((j / amount) * 100.00):.02f}% done. Training on {file}...")
                    for i, line in enumerate(open(file_path, 'r', encoding=LARGE_INPUT_FILE_ENCODING)):
                        if i == 0:
                            continue
                        elif line == '':
                            break
                        delim_i = line.find(',')
                        yield utils.simple_preprocess(line[delim_i + 1:])
        else:
            print(f"File path given. Generating reviews from {self.__path}")
            for i, line in enumerate(open(self.__path, 'r', encoding=LARGE_INPUT_FILE_ENCODING)):
                if i == 0:
                    continue
                delim_i = line.find(',')
                yield utils.simple_preprocess(line[delim_i + 1:])
        print("Gone over all files!")


# def load_data(path):
#     # this code loads a csv with strings into a pandas dataframe
#     df = pd.read_csv(path, encoding=LARGE_INPUT_FILE_ENCODING, sep=',', header=0, skipinitialspace=True,
#                      quotechar='"', escapechar='\\', nrows=1000)
#     # shuffle data and make separator
#     df = df.sample(frac=1).reset_index(drop=True)
#     sep = int(df.shape[0] * .75)
#
#     # train set, test set
#     return df.iloc[: sep, :], df.iloc[sep:, :]


def save_model(path, model, n=0):
    """
    Saves the given model in the 'textVectorizationModels' directory
    :param path: path to save model to
    :param model: model to be saved
    :param n: (optional) serial number to add to the file name
    """
    print(f"Saving to {path}...")
    save_path = os.path.join('..', os.path.join('textVectorizationModels',
                                                f"{os.path.basename(path).split('.')[0]}_{n}.model"))
    model.save(save_path)
    print("Done!")


def load_model(path, local=True):
    """
    Loads a pre-trained model either from a local directory or from the gensim api
    :param path: path to local model or name of gensim pre-trained model to download
    :param local: (bool) whether the model is stored locally or not
    :return: loaded model
    """
    print(f"Loading model from {path}...")
    if local:
        return gensim.models.KeyedVectors.load(path)
    return api.load(path)


def main(path_to_data, load=False, local=True, limit=0, save=True):
    if load:
        load_path = os.path.join(r'../textVectorizationModels', f"{path_to_data}.model")
        model = load_model(load_path, local)
        # print(model.wv.doesnt_match(['favorite', 'best', 'good', 'bad']))
        # print(model.similarity('best', 'good'))
        # print(model.most_similar('shock'))
        # print(model.wv.similarity('excellent', 'great'))
        pos = ['king', 'house']
        neg = []
        res1 = model.wv.most_similar_cosmul(positive=pos, negative=neg)
        res2 = model.wv.most_similar(positive=pos, negative=neg)
        print("%s: %.4f\n%s: %.4f" % (*res1[0], *res2[0]))
    else:
        if local:
            sentences = TextGenerator(path_to_data, limit=limit)
        else:
            sentences = api.load(path_to_data)
        model = Word2Vec(sentences=sentences, size=WORD_SPACE_DIM, workers=4)
    if save:
        save_model(path_to_data, model, n=limit)


if __name__ == '__main__':
    # if not 3 <= len(sys.argv) <= 4:
    #     print('Usage: python model <path-to-data-or-model> -n <optional-file-limit>')
    #     exit(-1)
    main(r"csv_80", load=True, save=False)
