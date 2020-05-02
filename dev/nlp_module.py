import sys
import numpy as np
from typing import List, Any
from textblob import TextBlob
from flair.data import Sentence
from flair.models import TextClassifier
from spellchecker import SpellChecker
from nltk import pos_tag, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from text_vectorization import W2v, T2v


MODIFIERS = {0: 'terrible', 1: 'excellent'}
ALLOWED_TAGS = {'JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS', 'VBG'}


def extract_features(review: str, t2v: T2v, w2v: W2v):
    # evaluate this review
    feature_scores = t2v.transform([review]).flatten()
    # remove empty values and get the sorting by maximum score first
    indices = np.argsort(feature_scores)[:: -1]
    feature_scores = np.sort(feature_scores)[:: -1]
    nonzero_indices = np.nonzero(feature_scores)[0]
    sorting = indices[nonzero_indices]
    # get the features corresponding to the scores sorted from best to worst
    features = np.array(t2v.get_feature_names())[sorting]

    # return synonyms for each feature
    syns = []
    splchkr = SpellChecker()
    tag_dict = {w[0]: w[1]
                for w in pos_tag(word_tokenize(review))
                if w[1] in ALLOWED_TAGS and w[0].lower() in features}
    # if none of the features found are in the allowed tags - attempt to use them anyway
    if not len(tag_dict):
        tag_dict = {w[0]: w[1]
                    for w in pos_tag(word_tokenize(review))
                    if w[0].lower() in features}

    used_words = set([f for f in tag_dict.keys()])
    for f in tag_dict.keys():
        # create list to hold feature in the middle and synonyms from each side
        curr = list([0] * (len(MODIFIERS) + 1))
        # save with capitalization if it exists!!!
        curr[(len(curr) // 2)] = f
        # convert to lower case for convenience
        f = f.lower()
        for i in range(len(MODIFIERS)):
            pos = [f, MODIFIERS[i]]
            neg = [MODIFIERS[len(MODIFIERS) - 1 - i]]
            # get similarity list
            sim_list = list(map(lambda w: w[0], w2v.wv.most_similar_cosmul(positive=pos, negative=neg)))
            # filter list from used words, non-adjectives and correct spelling errors
            sim_list = [splchkr.correction(w[0])
                        for w in pos_tag(sim_list)
                        if w[1] in ALLOWED_TAGS and w[0] not in used_words]
            # take the most similar
            # skip the cell with the feature itself
            curr[i if i < len(curr) // 2 else i + 1] = sim_list[0] \
                if curr[len(curr) // 2][0].islower() \
                else sim_list[0][0].upper() + sim_list[0][1:]

            used_words.add(sim_list[0])
        syns.append(curr)
    return syns


def combined_classification(review: str) -> List[Any]:
    confidence = 0.0
    # flair classification
    flair = TextClassifier.load('en-sentiment')  # type: TextClassifier
    as_sentence = Sentence(review)
    flair.predict(as_sentence)
    confidence += (.9 / 2.1) * as_sentence.labels[0].score
    # vader classification
    vader = SentimentIntensityAnalyzer()
    confidence += (.6 / 2.1) * vader.polarity_scores(review)['compound']
    # textblob classification
    confidence += (.6 / 2.1) * TextBlob(review).sentiment.polarity
    return ["POSITIVE" if confidence > 0 else "NEGATIVE", confidence]


def main(review: str):
    print(f"Analyzing: {review}")
    w2v = W2v(path="w2v_csv_80.model", load=True)
    t2v = T2v(name="t2v_50k.model")
    print(combined_classification(review))
    print(extract_features(review, t2v, w2v))


if __name__ == '__main__':
    if len(sys.argv) != 2 or len(sys.argv[1]) == 0:
        print("Usage: nlp_module.py <review: str>")
        exit(-1)
    main(sys.argv[1])


